# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import argparse
import copy
import glob
import logging
import os
import pickle
import random

import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import ale_py
import gymnasium as gym

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from alphaex.sweeper import Sweeper
from pearl import action_representation_modules
from pearl.action_representation_modules import IdentityActionRepresentationModule
from pearl.neural_networks.common import value_networks
from pearl.neural_networks.common.utils import (
    kaiming_normal_init_weights,
    orthogonal_init_weights,
    xavier_init_weights,
)
from pearl.neural_networks.sequential_decision_making import (
    actor_networks,
    q_value_networks,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners import sequential_decision_making as policy_learners
from pearl.policy_learners.exploration_modules import (
    common as exploration_modules,
    wrappers as exploration_wrappers,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.replay_buffers import sequential_decision_making as replay_buffers

from pearl.user_envs.wrappers import (
    AgentResetWrapper,
    EpisodicLifeEnv,
    FireResetEnv,
    HalfCheetahWrapper,
    MaxAndSkipEnv,
    NoopResetEnv,
    PusherWrapper,
    ReacherWrapper,
    ResetWrapper,
    SwimmerWrapper,
)
from pearl.utils.functional_utils.learning.preprocessing import (
    ObservationNormalization,
    Preprocessor,
    RewardClipping,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC

from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces import BoxActionSpace, DiscreteActionSpace


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_env(env_config: Dict[str, Any]) -> GymEnvironment:
    """
    attach a versatility wrapper to the environment
    """
    env_config = env_config[0]

    if env_config.get("reset_cost_wrapper", False):
        return GymEnvironment(
            ResetWrapper(
                env=get_gym_env(env_config),
                random_reset_prob=env_config.get("random_reset_prob", None),
                reset_cost=env_config.get("reset_cost", None),
                is_continuing=env_config.get("is_continuing", None),
            )
        )

    if env_config.get("agent_reset_cost_wrapper", False):
        return GymEnvironment(
            AgentResetWrapper(
                env=get_gym_env(env_config),
                reset_cost=env_config.get("reset_cost", None),
            )
        )

    return GymEnvironment(
        env_or_env_name=get_gym_env(env_config),
    )


def env_supports_termination_when_unhealthy(env_name: str) -> bool:
    return (
        "Ant-" in env_name
        or "AntNew-" in env_name
        or "SpecialAnt-" in env_name
        or "Hopper-" in env_name
        or "Walker2d-" in env_name
        or "Humanoid-" in env_name
    )


def get_gym_env(env_config: Dict) -> gym.Env:
    env_name = env_config.get("env_name", None)
    if env_name is None:
        raise ValueError("env_name is not specified")
    render_mode = env_config.get("render_mode", None)
    terminate_when_unhealthy = env_config.get("terminate_when_unhealthy", None)
    if env_supports_termination_when_unhealthy(env_name) is True:
        assert (
            terminate_when_unhealthy is not None
        ), f"terminate_when_unhealthy should be specified for {env_name}"

    if "AntNew-" in env_name:
        return gym.make(
            env_name.replace("AntNew-", "Ant-"),
            render_mode=render_mode,
            healthy_z_range=(0.3, 1.0),
            terminate_when_unhealthy=terminate_when_unhealthy,
        )
    elif "HalfCheetahNew-" in env_name:
        return HalfCheetahWrapper(
            gym.make(
                env_name.replace("HalfCheetahNew-", "HalfCheetah-"),
                render_mode=render_mode,
            )
        )
    elif "ReacherNew-" in env_name:
        return ReacherWrapper(
            gym.make(
                env_name.replace("ReacherNew-", "Reacher-"),
                render_mode=render_mode,
            )
        )
    elif "PusherNew-" in env_name:
        return PusherWrapper(
            gym.make(
                env_name.replace("PusherNew-", "Pusher-"),
                render_mode=render_mode,
            )
        )
    elif "SwimmerNew-" in env_name:
        return SwimmerWrapper(
            gym.make(
                env_name.replace("SwimmerNew-", "Swimmer-"),
                render_mode=render_mode,
            )
        )
    elif "SpecialAnt-" in env_name:
        return gym.make(
            env_name.replace("SpecialAnt-", "Ant-"),
            xml_file="special_ant.xml",
            render_mode=render_mode,
            terminate_when_unhealthy=terminate_when_unhealthy,
        )
    elif "ALE/" in env_name or "NoFrameskip" in env_name:
        # Atari envs
        max_num_frames_per_episode = env_config.get("max_num_frames_per_episode", None)
        if max_num_frames_per_episode is not None:
            max_num_frames_per_episode = int(max_num_frames_per_episode)
        env = gym.make(
            env_name,
            render_mode=render_mode,
            max_num_frames_per_episode=max_num_frames_per_episode,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
    else:
        if env_supports_termination_when_unhealthy(env_name) is True:
            return gym.make(
                env_name,
                render_mode=render_mode,
                terminate_when_unhealthy=terminate_when_unhealthy,
            )
        else:
            return gym.make(
                env_name,
                render_mode=render_mode,
            )


def generate_video(imgs, save_folder, video_name) -> None:
    """
    create a video using images from imgs.
    imgs: a list of np arrays representing a list of images
    save_folder: the folder to save the video and the generated images
    env_name: used as file name
    agent_name: used_as file name
    """
    # First create and save png files for imgs.
    for i in range(len(imgs)):
        logger.info(f"save image {i}")
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.savefig(f"{save_folder}/{video_name}%02d.png" % i)

    # Create a video using the generated png files.
    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            "16",
            "-i",
            f"{save_folder}/{video_name}%02d.png",
            # "-r",
            # "60",
            "-pix_fmt",
            "yuv420p",
            f"{save_folder}/{video_name}.mp4",
        ]
    )

    # Remove the generated png files.
    for file_name in glob.glob(f"{save_folder}/{video_name}*.png"):
        os.remove(file_name)


def evaluate(
    agent,
    env,
    output_dir,
    video_dir,
    name,
    render,
    max_steps,
    run_idx,
    preprocessors,
    qpos=None,  # for mujoco env
    qvel=None,  # for mujoco env
) -> None:

    # evaluate the agent in the continuing version of the environment
    logger.info(f"evaluating {name}, run {run_idx}")
    if hasattr(agent.policy_learner.exploration_module, "set_test_time_true"):
        # Do not change counter in the exploration module during evaluation
        agent.policy_learner.exploration_module.set_test_time_true()
    if hasattr(agent.policy_learner, "_test_time"):
        agent.policy_learner._test_time = True
    for p in preprocessors:
        if hasattr(p, "_test_time"):
            p._test_time = True

    observation, action_space = env.reset()
    if qpos is not None and qvel is not None:
        env.env.set_state(qpos, qvel)
    print_every_x_steps = 1000
    last_cum_reward_print = 0
    last_cum_reset_print = 0
    eval_cum_reward = 0
    eval_cum_reward_no_reset_cost = 0
    eval_cum_reset = 0
    frames = []
    agent.reset(observation, action_space)

    for step in range(1, max_steps + 1):
        if render is True:
            frames.append(env.env.render())
        action = agent.act(exploit=False)
        action = (
            action.cpu() if isinstance(action, torch.Tensor) else action
        )  # action can be int sometimes
        action_result = env.step(action)
        for preprocessor in preprocessors:
            preprocessor.process(action_result)
        eval_cum_reward += action_result.reward
        if "reset" in action_result.info:
            eval_cum_reset += action_result.info["reset"]
        if action_result.info["reset"] == False:
            eval_cum_reward_no_reset_cost += action_result.reward
        agent.observe(action_result)
        if step % print_every_x_steps == 0:
            logger.info(
                f"step {step}, agent={agent}, name={name}, average_reward={(eval_cum_reward - last_cum_reward_print) / print_every_x_steps}, average_reset={(eval_cum_reset - last_cum_reset_print) / print_every_x_steps}",
            )
            last_cum_reward_print = eval_cum_reward
            last_cum_reset_print = eval_cum_reset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if render is True:
        generate_video(frames, video_dir, video_name=name + "_" + str(run_idx))
    logger.info(f"{output_dir}/{run_idx}_final.npy")
    np.save(
        f"{output_dir}/{run_idx}_final.npy",
        np.array(
            [
                eval_cum_reward / max_steps,
                eval_cum_reward_no_reset_cost / max_steps,
                eval_cum_reset,
            ]
        ),
    )
    return None


def offline_eval(
    eval_agent,
    envs,
    eval_max_steps,
    preprocessors,
    eval_in_episodic_env=False,
    eval_in_continuing_env=False,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    agent = eval_agent
    if hasattr(agent.policy_learner.exploration_module, "set_test_time_true"):
        # Do not change counter in the exploration module during evaluation
        agent.policy_learner.exploration_module.set_test_time_true()
    if hasattr(agent.policy_learner, "_test_time"):
        agent.policy_learner._test_time = True
    for p in preprocessors:
        if hasattr(p, "_test_time"):
            p._test_time = True
    if eval_in_episodic_env:
        assert envs[1] is not None
        # evaluate the agent in the episodic version of the environment
        eps_return_list = []
        eps_clipped_return_list = []
        episode_steps = 0
        total_steps = 0
        eps_return = 0

        while total_steps < eval_max_steps:
            info, episode_steps = run_episode(
                agent=agent,
                env=envs[1],
                exploit=False,
                learn=False,
                preprocessors=preprocessors,
                total_steps=total_steps,
                number_of_steps=eval_max_steps,
            )
            total_steps += episode_steps
            eps_return_list.append(info["return"])
            if "clipped_return" in info:
                eps_clipped_return_list.append(info["clipped_return"])
        eps_return = np.mean(eps_return_list)
        if len(eps_clipped_return_list) > 0:
            eps_clipped_return = np.mean(eps_clipped_return_list)
        else:
            eps_clipped_return = None
    else:
        eps_return = None
        eps_clipped_return = None

    if eval_in_continuing_env:
        assert envs[2] is not None
        # evaluate the agent in the continuing version of the environment
        observation, action_space = envs[2].reset()
        agent.reset(observation, action_space)
        eval_cum_reward = 0
        use_reward_clipping = False
        for preprocessor in preprocessors:
            if isinstance(preprocessor, RewardClipping):
                use_reward_clipping = True
                break
        if use_reward_clipping:
            eval_cum_clipped_reward = 0
        else:
            eval_cum_clipped_reward = None
        for _ in range(1, eval_max_steps + 1):
            action = agent.act(exploit=False)
            action = (
                action.cpu() if isinstance(action, torch.Tensor) else action
            )  # action can be int sometimes
            action_result = envs[2].step(action)
            eval_cum_reward += action_result.reward
            for preprocessor in preprocessors:
                preprocessor.process(action_result)
            if use_reward_clipping:
                eval_cum_clipped_reward += action_result.reward
            agent.observe(action_result)
        avg_reward = eval_cum_reward / eval_max_steps
        if eval_cum_clipped_reward is not None:
            avg_clipped_reward = eval_cum_clipped_reward / eval_max_steps
        else:
            avg_clipped_reward = None
    else:
        avg_reward = None
        avg_clipped_reward = None
    logger.info(f"offline eval, epsodic return {eps_return}, avg reward {avg_reward}")
    if hasattr(agent.policy_learner.exploration_module, "set_test_time_false"):
        # Do not change counter in the exploration module during evaluation
        agent.policy_learner.exploration_module.set_test_time_false()
    if hasattr(agent.policy_learner, "_test_time"):
        agent.policy_learner._test_time = False
    for p in preprocessors:
        if hasattr(p, "_test_time"):
            p._test_time = False
    return eps_return, avg_reward, eps_clipped_return, avg_clipped_reward


def run_episode(
    agent: PearlAgent,
    env,
    exploit: bool = True,
    learn_after_episode: bool = False,
    learn_every_k_steps: int = 1,
    total_steps: int = 0,
    seed=None,
    learn: bool = True,
    learning_start: int = 0,
    preprocessors: Optional[List[Preprocessor]] = None,
    learning_report_cache: Optional[Dict[str, List[float]]] = None,
    number_of_steps: Optional[int] = None,
    visited_observations: Optional[List[Any]] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Runs one episode and returns an info dict and number of steps taken.

    Args:
        agent (Agent): the agent.
        env (Environment): the environment.
        learn (bool, optional): Runs `agent.learn()` after every step. Defaults to False.
        exploit (bool, optional): asks the agent to only exploit. Defaults to False.
        learn_after_episode (bool, optional): asks the agent to only learn at
                                              the end of the episode. Defaults to False.
        learn_every_k_steps (int, optional): asks the agent to learn every k steps.
        total_steps (int, optional): the total number of steps taken so far. Defaults to 0.
        seed (int, optional): the seed for the environment. Defaults to None.
    Returns:
        Tuple[Dict[str, Any], int]: the return of the episode and the number of steps taken.
    """
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    cum_reward = 0
    use_reward_clipping = False
    for preprocessor in preprocessors:
        if isinstance(preprocessor, RewardClipping):
            use_reward_clipping = True
            break
    if use_reward_clipping:
        cum_clipped_reward = 0
    done = False
    episode_steps = 0
    info = {}

    while not done:
        action = agent.act(exploit=exploit)
        action = (
            action.cpu() if isinstance(action, torch.Tensor) else action
        )  # action can be int sometimes
        action_result = env.step(action)
        if visited_observations is not None and total_steps + episode_steps % 1000 == 0:
            visited_observations.append(action_result.observation)
        if "episode" in action_result.info:
            logger.info(action_result.info["episode"]["r"])
            info["full_return"] = action_result.info["episode"]["r"]
        cum_reward += action_result.reward
        for preprocessor in preprocessors:
            preprocessor.process(action_result)
        if use_reward_clipping:
            cum_clipped_reward += action_result.reward
        agent.observe(action_result)
        done = action_result.truncated or action_result.terminated
        episode_steps += 1
        if learn and total_steps + episode_steps >= learning_start:
            if learn_after_episode:
                # when learn_after_episode is True, we learn only at the end of the episode,
                # regardless of the value of learn_every_k_steps.
                if done:
                    report = agent.learn()
                    for key in report:
                        learning_report_cache.setdefault(key, []).append(report[key])
            else:
                assert learn_every_k_steps > 0, "learn_every_k_steps must be positive"
                if (total_steps + episode_steps) % learn_every_k_steps == 0:
                    report = agent.learn()
                    for key in report:
                        learning_report_cache.setdefault(key, []).append(report[key])
        if (
            number_of_steps is not None
            and total_steps + episode_steps >= number_of_steps
        ):
            break

    info["return"] = cum_reward
    if use_reward_clipping:
        info["clipped_return"] = cum_clipped_reward

    return info, episode_steps


def run_episodes(train_agent, eval_agent, envs, param_sweeper_dict) -> None:
    print_every_x_steps = param_sweeper_dict["print_every_x_steps"]
    learn_every_k_steps = param_sweeper_dict["learn_every_k_steps"]
    record_period = param_sweeper_dict["record_period"]
    run_idx = param_sweeper_dict["id"]
    number_of_steps = param_sweeper_dict["max_steps"]
    eval_max_steps = param_sweeper_dict["eval_max_steps"]
    total_steps = 0
    total_episodes = 0
    info = {}
    info_period = {}
    eval_episodic_return_list, eval_average_reward_list, eval_episodic_clipped_return_list, eval_average_clipped_reward_list = [], [], [], []  # noqa
    agent = train_agent
    env = envs[0]  # train env
    learning_report = {}
    learning_report_cache = {}
    start_time = time.time()
    last_timed_steps = 0
    if param_sweeper_dict.get("record_visited_observations", False) is True:
        visited_observations = []
    else:
        visited_observations = None
    while True:
        if total_steps >= number_of_steps:
            break
        old_total_steps = total_steps
        episode_info, episode_total_steps = run_episode(
            agent,
            env,
            exploit=False,
            learn_after_episode=False,  # not for this project
            learn_every_k_steps=learn_every_k_steps,
            total_steps=old_total_steps,
            learning_start=param_sweeper_dict["learning_starts"],
            learn=True,
            preprocessors=param_sweeper_dict["preprocessors"],
            learning_report_cache=learning_report_cache,
            number_of_steps=number_of_steps,
            visited_observations=visited_observations,
        )
        total_steps += episode_total_steps
        total_episodes += 1
        if old_total_steps // print_every_x_steps < total_steps // print_every_x_steps:
            logger.info(
                f"episode {total_episodes}, steps {total_steps}, agent={agent}, env={env}",
            )
            end_time = time.time()
            SPS = int((total_steps - last_timed_steps) / (end_time - start_time))
            logger.info(f"samples per second: {SPS}")
            start_time = end_time
            last_timed_steps = total_steps
            for key in episode_info:
                logger.info(f"{key}: {episode_info[key]}")

        for key in episode_info:
            info_period.setdefault(key, []).append(episode_info[key])
        if old_total_steps // record_period < total_steps // record_period:

            # record average info value every record_period steps
            num_repeating_recordings = (total_steps // record_period) - (
                old_total_steps // record_period
            )
            for _ in range(num_repeating_recordings):
                for key in info_period:
                    info.setdefault(key, []).append(np.mean(info_period[key]))
            info_period = {}
            # evaluate the learned policy in the episodic and continuing versions of the environment
            eps_return, avg_reward, eps_clipped_return, average_clipped_reward = (
                offline_eval(
                    eval_agent=eval_agent,
                    envs=envs,
                    eval_max_steps=eval_max_steps,
                    preprocessors=param_sweeper_dict["preprocessors"],
                    eval_in_episodic_env=param_sweeper_dict["eval_in_episodic_env"],
                    eval_in_continuing_env=param_sweeper_dict["eval_in_continuing_env"],
                )
            )
            for _ in range(num_repeating_recordings):
                if eps_return is not None:
                    eval_episodic_return_list.append(eps_return)
                if avg_reward is not None:
                    eval_average_reward_list.append(avg_reward)
                if eps_clipped_return is not None:
                    eval_episodic_clipped_return_list.append(eps_clipped_return)
                if average_clipped_reward is not None:
                    eval_average_clipped_reward_list.append(average_clipped_reward)
                for key in learning_report_cache:
                    learning_report.setdefault(key, []).append(
                        np.mean(learning_report_cache[key])
                    )

    output_dir = param_sweeper_dict["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    for key in info:
        if key == "return":
            np.save(f"{output_dir}/{run_idx}_episodic_return.npy", info[key])
        if key == "clipped_return":
            np.save(f"{output_dir}/{run_idx}_episodic_clipped_return.npy", info[key])
        if key == "full_return":
            np.save(f"{output_dir}/{run_idx}_episodic_full_return.npy", info[key])
    if len(eval_episodic_return_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_episodic_return.npy",
            np.array(eval_episodic_return_list),
        )
    if len(eval_average_reward_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_average_reward.npy",
            np.array(eval_average_reward_list),
        )

    if len(eval_episodic_clipped_return_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_episodic_clipped_return.npy",
            np.array(eval_episodic_clipped_return_list),
        )
    if len(eval_average_clipped_reward_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_average_clipped_reward.npy",
            np.array(eval_average_clipped_reward_list),
        )

    for key in learning_report:
        np.save(
            f"{output_dir}/{run_idx}_{key}.npy",
            np.array(learning_report[key]),
        )
    if param_sweeper_dict.get("record_visited_observations", False) is True:
        np.save(
            f"{output_dir}/{run_idx}_visited_observations.npy",
            np.array(visited_observations),
        )


def run_steps(train_agent, eval_agent, envs, param_sweeper_dict) -> None:
    print_every_x_steps = param_sweeper_dict["print_every_x_steps"]
    learn_every_k_steps = param_sweeper_dict["learn_every_k_steps"]
    record_period = param_sweeper_dict["record_period"]
    run_idx = param_sweeper_dict["id"]
    max_steps = param_sweeper_dict["max_steps"]
    eval_max_steps = param_sweeper_dict["eval_max_steps"]
    env = envs[0]  # train env
    agent = train_agent
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    cum_reward = 0
    avg_reward_list = []
    eval_average_reward_list = []  # used only when eval_env_continuing is not None
    eval_episodic_return_list = []  # used only when eval_env_episodic is not None
    eval_average_clipped_reward_list = []
    eval_episodic_clipped_return_list = []
    last_cum_reward_print = cum_reward
    last_cum_reward_record = cum_reward
    learning_report = {}
    learning_report_cache = {}
    start_time = time.time()
    last_timed_steps = 0
    steps = 0
    use_reward_clipping = False
    if param_sweeper_dict.get("record_visited_observations", False) is True:
        visited_observations = []
    else:
        visited_observations = None

    for preprocessor in param_sweeper_dict["preprocessors"]:
        if isinstance(preprocessor, RewardClipping):
            use_reward_clipping = True
            break
    if use_reward_clipping:
        cum_clipped_reward = 0
        last_cum_clipped_reward_print = cum_reward
        last_cum_clipped_reward_record = cum_reward
        avg_clipped_reward_list = []
    while steps < max_steps:
        action = agent.act(exploit=False)
        action = (
            action.cpu() if isinstance(action, torch.Tensor) else action
        )  # action can be int sometimes
        action_result = env.step(action)
        if "reward_offset" in param_sweeper_dict:
            action_result.reward += param_sweeper_dict["reward_offset"]
        assert action_result.truncated is False

        cum_reward += action_result.reward
        for preprocessor in param_sweeper_dict["preprocessors"]:
            preprocessor.process(action_result)
        if use_reward_clipping:
            cum_clipped_reward += action_result.reward
        if visited_observations is not None and steps % 1000 == 0:
            visited_observations.append(action_result.observation)

        agent.observe(action_result)
        steps += 1
        if steps % print_every_x_steps == 0:
            if use_reward_clipping:
                logger.info(
                    f"steps {steps}, agent={agent}, env={env}, average_reward={(cum_reward - last_cum_reward_print) / print_every_x_steps}, average_clipped_reward={(cum_clipped_reward - last_cum_clipped_reward_print) / print_every_x_steps}",
                )
                last_cum_clipped_reward_print = cum_clipped_reward
            else:
                logger.info(
                    f"steps {steps}, agent={agent}, env={env}, average_reward={(cum_reward - last_cum_reward_print) / print_every_x_steps}",
                )
            last_cum_reward_print = cum_reward
            end_time = time.time()
            SPS = int((steps - last_timed_steps) / (end_time - start_time))
            logger.info(f"samples per second: {SPS}")
            start_time = end_time
            last_timed_steps = steps
        if steps % record_period == 0:
            # record the average reward over the last record_period time steps
            avg_reward_list.append(
                (cum_reward - last_cum_reward_record) / record_period
            )
            last_cum_reward_record = cum_reward
            if use_reward_clipping:
                avg_clipped_reward_list.append(
                    (cum_clipped_reward - last_cum_clipped_reward_record)
                    / record_period
                )
                last_cum_clipped_reward_record = cum_clipped_reward
            # evaluate the learned policy in the episodic and continuing versions of the environment
            eps_return, avg_reward, eps_clipped_return, avg_clipped_reward = (
                offline_eval(
                    eval_agent=eval_agent,
                    envs=envs,
                    eval_max_steps=eval_max_steps,
                    preprocessors=param_sweeper_dict["preprocessors"],
                    eval_in_episodic_env=param_sweeper_dict["eval_in_episodic_env"],
                    eval_in_continuing_env=param_sweeper_dict["eval_in_continuing_env"],
                )
            )
            if eps_return is not None:
                eval_episodic_return_list.append(eps_return)
            if avg_reward is not None:
                eval_average_reward_list.append(avg_reward)
            if eps_clipped_return is not None:
                eval_episodic_clipped_return_list.append(eps_clipped_return)
            if avg_clipped_reward is not None:
                eval_average_clipped_reward_list.append(avg_clipped_reward)
            for key in learning_report_cache:
                learning_report.setdefault(key, []).append(
                    np.mean(learning_report_cache[key])
                )

        assert learn_every_k_steps > 0, "learn_every_k_steps must be positive"
        if (
            steps >= param_sweeper_dict["learning_starts"]
            and steps % learn_every_k_steps == 0
        ):
            report = agent.learn()
            for key in report:
                learning_report_cache.setdefault(key, []).append(report[key])
    output_dir = param_sweeper_dict["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    np.save(
        f"{output_dir}/{run_idx}_average_reward.npy",
        np.array(avg_reward_list),
    )
    if use_reward_clipping:
        np.save(
            f"{output_dir}/{run_idx}_average_clipped_reward.npy",
            np.array(avg_clipped_reward_list),
        )
    if len(eval_episodic_return_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_episodic_return.npy",
            np.array(eval_episodic_return_list),
        )
    if len(eval_average_reward_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_average_reward.npy",
            np.array(eval_average_reward_list),
        )
    if len(eval_episodic_clipped_return_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_episodic_clipped_return.npy",
            np.array(eval_episodic_clipped_return_list),
        )
    if len(eval_average_clipped_reward_list) > 0:
        np.save(
            f"{output_dir}/{run_idx}_eval_average_clipped_reward.npy",
            np.array(eval_average_clipped_reward_list),
        )
    for key in learning_report:
        np.save(
            f"{output_dir}/{run_idx}_{key}.npy",
            np.array(learning_report[key]),
        )
    if param_sweeper_dict.get("record_visited_observations", False) is True:
        np.save(
            f"{output_dir}/{run_idx}_visited_observations.npy",
            np.array(visited_observations),
        )
    # save game states so that we can visualize the behavior of the agent after training
    # by running the learned policy starting from the saved game state
    if (
        "save_last_state" in param_sweeper_dict
        and param_sweeper_dict["save_last_state"]
    ):
        if (
            hasattr(env.env, "data")
            and hasattr(env.env.data, "qpos")
            and hasattr(env.env.data, "qvel")
        ):
            # This is the case for mujoco games
            np.save(f"{output_dir}/{run_idx}_qpos.npy", np.array(env.env.data.qpos))
            np.save(f"{output_dir}/{run_idx}_qvel.npy", np.array(env.env.data.qvel))
        else:
            raise NotImplementedError("Can not obtain last environment state!")
    return


def init_class(module_class, module_name, param_sweeper_dict) -> None:
    filtered_dict = {}
    for key, value in param_sweeper_dict.items():
        prefix = module_name + ":"
        if (
            len(key) > len(prefix)
            and prefix == key[0 : len(prefix)]
            and key[len(prefix) :] != "type"
        ):
            filtered_dict[key[len(prefix) :]] = value
    # logger.info(filtered_dict)
    param_sweeper_dict[module_name] = module_class(**filtered_dict)


def ppo_init_network_continuous(param_sweeper_dict) -> None:
    param_sweeper_dict["actor_network_instance"].apply(orthogonal_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(orthogonal_init_weights)
    if hasattr(param_sweeper_dict["actor_network_instance"], "fc_mu"):
        param_sweeper_dict["actor_network_instance"].fc_mu.weight.data.copy_(
            0.01 * param_sweeper_dict["actor_network_instance"].fc_mu.weight.data
        )


def ppo_init_network_discrete(param_sweeper_dict) -> None:
    param_sweeper_dict["actor_network_instance"].apply(orthogonal_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(orthogonal_init_weights)
    if hasattr(param_sweeper_dict["actor_network_instance"], "_model_fc"):
        param_sweeper_dict["actor_network_instance"]._model_fc[-1][0].weight.data.copy_(
            0.01
            / 1.4142
            * param_sweeper_dict["actor_network_instance"]._model_fc[-1][0].weight.data
        )
    if hasattr(param_sweeper_dict["critic_network_instance"], "_model_fc"):
        param_sweeper_dict["critic_network_instance"]._model_fc[-1][
            0
        ].weight.data.copy_(
            1.0
            / 1.4142
            * param_sweeper_dict["critic_network_instance"]._model_fc[-1][0].weight.data
        )


def sac_atari_init_network(param_sweeper_dict) -> None:
    param_sweeper_dict["actor_network_instance"].apply(kaiming_normal_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(kaiming_normal_init_weights)


def ac_init_network(param_sweeper_dict) -> None:
    param_sweeper_dict["actor_network_instance"].apply(xavier_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(xavier_init_weights)


def q_init_network(param_sweeper_dict) -> None:
    param_sweeper_dict["network_instance"].apply(xavier_init_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--gpu-id", default=-1)
    parser.add_argument("--base-id", default=0)
    parser.add_argument(
        "--config-file", default="experiments/refactor/no_resets_mujoco/inputs.json"
    )
    parser.add_argument("--out-dir", default="/tmp/pearl")
    parser.add_argument("--eval-agent", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    exp_name = args.config_file.split("/")[1]
    project_root = os.path.abspath(os.path.dirname(__file__))
    param_sweeper = Sweeper(os.path.join(project_root, args.config_file))
    envs_configs = ["env", "eval_env_episodic", "eval_env_continuing"]
    agents = []
    envs = []
    run_id = int(args.base_id)
    param_sweeper_dict = param_sweeper.parse(run_id)
    param_sweeper_dict["id"] = run_id
    param_sweeper_dict["device_id"] = args.gpu_id
    param_sweeper_dict["output_dir"] = args.out_dir

    """
    log hyper parameters
    """

    for item in param_sweeper_dict:
        logger.info(f"{item}: {param_sweeper_dict[item]}")

    """
    set random seed
    """

    set_seed(run_id)

    """
    Initialize the environment
    """

    for i in range(3):
        # envs[0]: training agent and env
        # envs[1]: evaluated in episodic env
        # envs[2]: evaluated in continuing env
        if envs_configs[i] not in param_sweeper_dict:
            envs.append(None)
            continue
        if args.render:
            param_sweeper_dict[envs_configs[i]][0]["render_mode"] = "rgb_array"
            env = get_env(param_sweeper_dict[envs_configs[i]])
        else:
            env = get_env(param_sweeper_dict[envs_configs[i]])
        env.action_space._gym_space.seed(seed=run_id)
        env.reset(seed=run_id)
        envs.append(env)

    env = envs[0]  # training environment
    param_sweeper_dict["action_space"] = env.action_space
    param_sweeper_dict["preprocessors"] = []
    if isinstance(env.action_space, DiscreteActionSpace):
        max_number_actions = env.action_space.n
        action_dim = env.action_space.action_dim
    elif isinstance(env.action_space, BoxActionSpace):
        max_number_actions = -1
        action_dim = env.action_space.action_dim
    else:
        raise NotImplementedError

    """
    Initialize preprocessors
    """
    param_sweeper_dict["preprocessors"] = []
    training_env_name = param_sweeper_dict["env"][0]["env_name"]
    if "ALE/" in training_env_name or "NoFrameskip" in training_env_name:
        param_sweeper_dict["preprocessors"].append(RewardClipping())

    if (
        param_sweeper_dict["policy_learner:type"] == "ProximalPolicyOptimization"
        and param_sweeper_dict["is_action_continuous"] is True
    ):
        param_sweeper_dict["preprocessors"].append(
            ObservationNormalization(envs[0].observation_space.shape)
        )

    """
    Initialize action representation module
    """

    if "action_representation_module:type" in param_sweeper_dict:
        # if action representation module name is specified, initialize a module
        if param_sweeper_dict["action_representation_module:type"] in [
            "OneHotActionTensorRepresentationModule",
        ]:
            param_sweeper_dict["action_representation_module:max_number_actions"] = (
                max_number_actions
            )
        elif param_sweeper_dict["action_representation_module:type"] in [
            "IdentityActionRepresentationModule"
        ]:
            param_sweeper_dict["action_representation_module:representation_dim"] = (
                action_dim
            )
        else:
            raise NotImplementedError
        action_representation_module_class = getattr(
            action_representation_modules,
            param_sweeper_dict["action_representation_module:type"],
        )
        init_class(
            action_representation_module_class,
            "action_representation_module",
            param_sweeper_dict,
        )
    else:
        param_sweeper_dict["action_representation_module"] = (
            IdentityActionRepresentationModule(
                max_number_actions=max_number_actions,
                representation_dim=action_dim,
            )
        )

    action_representation_dim = param_sweeper_dict[
        "action_representation_module"
    ].representation_dim

    """
    Initialize exploration module
    """

    if "exploration_module:type" in param_sweeper_dict:
        # if exploration module name is specified, initialize an exploration module
        exploration_module_class = getattr(
            exploration_modules, param_sweeper_dict["exploration_module:type"]
        )
        if "exploration_module:std_dev" in param_sweeper_dict and isinstance(
            param_sweeper_dict["exploration_module:std_dev"], list
        ):
            assert isinstance(env.action_space, BoxActionSpace)
            assert len(param_sweeper_dict["exploration_module:std_dev"]) == 2
            tmp = (
                torch.ones(
                    param_sweeper_dict[
                        "action_representation_module:representation_dim"
                    ]
                )
                * param_sweeper_dict["exploration_module:std_dev"][0]
            )
            tmp[-1] = param_sweeper_dict["exploration_module:std_dev"][1]
            param_sweeper_dict["exploration_module:std_dev"] = tmp
        init_class(exploration_module_class, "exploration_module", param_sweeper_dict)
        if "exploration_module_wrapper:type" in param_sweeper_dict:
            # if exploration wrapper module name is specified, initialize an exploration module
            exploration_wrapper_class = getattr(
                exploration_wrappers,
                param_sweeper_dict["exploration_module_wrapper:type"],
            )
            param_sweeper_dict["exploration_module_wrapper:exploration_module"] = (
                param_sweeper_dict["exploration_module"]
            )
            init_class(
                exploration_wrapper_class,
                "exploration_module_wrapper",
                param_sweeper_dict,
            )
            param_sweeper_dict["exploration_module"] = param_sweeper_dict[
                "exploration_module_wrapper"
            ]
    else:
        param_sweeper_dict["exploration_module"] = NoExploration()

    if "actor_update_noise" in param_sweeper_dict and isinstance(
        param_sweeper_dict["actor_update_noise"], list
    ):
        assert isinstance(env.action_space, BoxActionSpace)
        assert len(param_sweeper_dict["actor_update_noise"]) == 2
        tmp = (
            torch.ones(
                param_sweeper_dict["action_representation_module:representation_dim"]
            )
            * param_sweeper_dict["actor_update_noise"][0]
        )
        tmp[-1] = param_sweeper_dict["actor_update_noise"][1]
        param_sweeper_dict["actor_update_noise"] = tmp

    """
    Initialize replay buffer
    """

    if "replay_buffer:type" in param_sweeper_dict:
        # if replay buffer is specified, intialize one
        replay_buffer_class = getattr(
            replay_buffers, param_sweeper_dict["replay_buffer:type"]
        )
        init_class(replay_buffer_class, "replay_buffer", param_sweeper_dict)
    else:
        print("Replay buffer must be specified")
        raise NotImplementedError

    """
    Initialize networks
    """

    if "network_instance:type" in param_sweeper_dict:
        # for q-learning methods
        if param_sweeper_dict["network_instance:type"] in [
            "CNNQValueNetwork",
            "CNNQValueMultiHeadNetwork",
        ]:
            # image based inputs
            assert len(env.observation_space.shape) == 3
            param_sweeper_dict["network_instance:input_width"] = 84
            param_sweeper_dict["network_instance:input_height"] = 84
            param_sweeper_dict["network_instance:input_channels_count"] = 4
        elif param_sweeper_dict["network_instance:type"] in [
            "VanillaQValueNetwork",
            "VanillaQValueMultiHeadNetwork",
        ]:
            # vector based inputs
            assert len(env.observation_space.shape) == 1
            param_sweeper_dict["network_instance:state_dim"] = (
                env.observation_space.shape[0]
            )
        else:
            raise NotImplementedError
        param_sweeper_dict["network_instance:action_dim"] = action_representation_dim
        network_class = getattr(
            q_value_networks, param_sweeper_dict["network_instance:type"]
        )
        init_class(network_class, "network_instance", param_sweeper_dict)

    if "actor_network_instance:type" in param_sweeper_dict:
        # if actor network is specified, intialize one
        if param_sweeper_dict["actor_network_instance:type"] in [
            "CNNActorNetwork",
        ]:
            # image based inputs
            assert len(env.observation_space.shape) == 3
            param_sweeper_dict["actor_network_instance:input_width"] = 84
            param_sweeper_dict["actor_network_instance:input_height"] = 84
            param_sweeper_dict["actor_network_instance:input_channels_count"] = 4
        elif param_sweeper_dict["actor_network_instance:type"] in [
            "VanillaActorNetwork",
            "VanillaContinuousActorNetwork",
            "GaussianActorNetwork",
            "ClipGaussianActorNetwork",
        ]:
            param_sweeper_dict["actor_network_instance:input_dim"] = (
                env.observation_space.shape[0]
            )
        else:
            raise NotImplementedError
        param_sweeper_dict["actor_network_instance:output_dim"] = (
            action_representation_dim
            if max_number_actions == -1  # continuous actions
            else max_number_actions  # discrete actions
        )
        param_sweeper_dict["actor_network_instance:action_space"] = env.action_space
        actor_class = getattr(
            actor_networks, param_sweeper_dict["actor_network_instance:type"]
        )
        init_class(actor_class, "actor_network_instance", param_sweeper_dict)

    if "critic_network_instance:type" in param_sweeper_dict:
        # if critic network is specified, intialize one
        if param_sweeper_dict["critic_network_instance:type"] in [
            "VanillaValueNetwork",
        ]:
            critic_class = getattr(
                value_networks, param_sweeper_dict["critic_network_instance:type"]
            )
            param_sweeper_dict["critic_network_instance:input_dim"] = (
                env.observation_space.shape[0]
            )
            init_class(critic_class, "critic_network_instance", param_sweeper_dict)
        elif param_sweeper_dict["critic_network_instance:type"] in [
            "CNNValueNetwork",
        ]:
            critic_class = getattr(
                value_networks, param_sweeper_dict["critic_network_instance:type"]
            )
            param_sweeper_dict["critic_network_instance:input_width"] = 84
            param_sweeper_dict["critic_network_instance:input_height"] = 84
            param_sweeper_dict["critic_network_instance:input_channels_count"] = 4
            init_class(critic_class, "critic_network_instance", param_sweeper_dict)
        elif param_sweeper_dict["critic_network_instance:type"] in [
            "EnsembleQValueNetwork",
        ]:
            list_of_member_networks = []
            ensemble_size = param_sweeper_dict["critic_network_instance:ensemble_size"]
            for _ in range(ensemble_size):
                if param_sweeper_dict["critic_member_network:type"] in [
                    "CNNQValueNetwork",
                    "CNNQValueMultiHeadNetwork",
                ]:
                    # image based inputs
                    assert len(env.observation_space.shape) == 3
                    param_sweeper_dict["critic_member_network:input_width"] = 84
                    param_sweeper_dict["critic_member_network:input_height"] = 84
                    param_sweeper_dict["critic_member_network:input_channels_count"] = 4
                elif param_sweeper_dict["critic_member_network:type"] in [
                    "VanillaQValueNetwork",
                    "VanillaQValueMultiHeadNetwork",
                ]:
                    # vector based inputs
                    assert len(env.observation_space.shape) == 1
                    param_sweeper_dict["critic_member_network:state_dim"] = (
                        env.observation_space.shape[0]
                    )
                else:
                    raise NotImplementedError
                param_sweeper_dict["critic_member_network:action_dim"] = (
                    action_representation_dim
                )
                member_network_class = getattr(
                    q_value_networks,
                    param_sweeper_dict["critic_member_network:type"],
                )
                filtered_dict = {}
                for key, value in param_sweeper_dict.items():
                    prefix = "critic_member_network:"
                    if (
                        len(key) > len(prefix)
                        and prefix == key[0 : len(prefix)]
                        and key[len(prefix) :] != "type"
                    ):
                        filtered_dict[key[len(prefix) :]] = value
                list_of_member_networks.append(member_network_class(**filtered_dict))
            models = nn.ModuleList(list_of_member_networks)
            critic_class = getattr(
                q_value_networks, param_sweeper_dict["critic_network_instance:type"]
            )
            param_sweeper_dict["critic_network_instance"] = critic_class(
                models=models, ensemble_size=ensemble_size
            )
        else:
            raise NotImplementedError

    if param_sweeper_dict.get("reward_centering:type", None) is not None:
        if param_sweeper_dict["reward_centering:type"] == "TD":
            param_sweeper_dict["reward_rate"] = nn.Parameter(torch.zeros(1))
        elif param_sweeper_dict["reward_centering:type"] == "MA":
            param_sweeper_dict["reward_rate"] = torch.zeros(1)
            param_sweeper_dict["reward_centering:ma_rate"] = param_sweeper_dict.get(
                "ma_rate", 0.99
            )
        elif param_sweeper_dict["reward_centering:type"] == "RVI":
            param_sweeper_dict["reward_rate"] = torch.zeros(1)
            param_sweeper_dict["reward_centering:ref_states_update_freq"] = (
                param_sweeper_dict.get(
                    "ref_states_update_freq", param_sweeper_dict["max_steps"]
                )
            )  # 0 means no update
        else:
            raise NotImplementedError
    else:
        param_sweeper_dict["reward_rate"] = torch.zeros(1)
    """
    network initialization
    """

    if (
        param_sweeper_dict["policy_learner:type"] == "ProximalPolicyOptimization"
        and param_sweeper_dict["is_action_continuous"] is True
    ):
        ppo_init_network_continuous(param_sweeper_dict)

    if (
        param_sweeper_dict["policy_learner:type"] == "ProximalPolicyOptimization"
        and param_sweeper_dict["is_action_continuous"] is False
    ):
        ppo_init_network_discrete(param_sweeper_dict)

    if param_sweeper_dict["policy_learner:type"] == "SoftActorCritic" and (
        "ALE/" in training_env_name or "NoFrameskip" in training_env_name
    ):
        sac_atari_init_network(param_sweeper_dict)

    """
    Initialize optimizers
    """

    if "optimizer:type" in param_sweeper_dict:
        assert "network_instance" in param_sweeper_dict
        optimizer_class = getattr(torch.optim, param_sweeper_dict["optimizer:type"])
        param_sweeper_dict["optimizer:params"] = param_sweeper_dict[
            "network_instance"
        ].parameters()
        init_class(optimizer_class, "optimizer", param_sweeper_dict)

    if "actor_optimizer:type" in param_sweeper_dict:
        assert "actor_network_instance" in param_sweeper_dict
        actor_optimizer_class = getattr(
            torch.optim, param_sweeper_dict["actor_optimizer:type"]
        )
        param_sweeper_dict["actor_optimizer:params"] = param_sweeper_dict[
            "actor_network_instance"
        ].parameters()
        init_class(actor_optimizer_class, "actor_optimizer", param_sweeper_dict)

    if "critic_optimizer:type" in param_sweeper_dict:
        assert "critic_network_instance" in param_sweeper_dict
        critic_optimizer_class = getattr(
            torch.optim, param_sweeper_dict["critic_optimizer:type"]
        )
        param_sweeper_dict["critic_optimizer:params"] = param_sweeper_dict[
            "critic_network_instance"
        ].parameters()
        init_class(critic_optimizer_class, "critic_optimizer", param_sweeper_dict)

    if param_sweeper_dict.get("reward_centering:type", None) is not None:
        if param_sweeper_dict["reward_centering:type"] == "TD":
            reward_rate_optimizer_class = getattr(
                torch.optim, param_sweeper_dict["reward_rate_optimizer:type"]
            )
            param_sweeper_dict["reward_rate_optimizer:params"] = [
                param_sweeper_dict["reward_rate"]
            ]
            init_class(
                reward_rate_optimizer_class, "reward_rate_optimizer", param_sweeper_dict
            )
            param_sweeper_dict["reward_centering:optimizer"] = param_sweeper_dict[
                "reward_rate_optimizer"
            ]
            init_class(TD_RC, "reward_centering", param_sweeper_dict)
        elif param_sweeper_dict["reward_centering:type"] == "MA":
            init_class(MA_RC, "reward_centering", param_sweeper_dict)
        elif param_sweeper_dict["reward_centering:type"] == "RVI":
            init_class(RVI_RC, "reward_centering", param_sweeper_dict)
    """
    Initialize a policy learner
    """

    policy_learner_class = getattr(
        policy_learners, param_sweeper_dict["policy_learner:type"]
    )

    filtered_dict = {
        key: value
        for key, value in param_sweeper_dict.items()
        if key in policy_learner_class.__init__.__code__.co_varnames
    }
    param_sweeper_dict["policy_learner"] = policy_learner_class(**filtered_dict)

    """
    Initialize a pearl agent
    """

    filtered_dict = {
        key: value
        for key, value in param_sweeper_dict.items()
        if key in PearlAgent.__init__.__code__.co_varnames
    }
    train_agent = PearlAgent(**filtered_dict)

    """
    Run the experiment
    """

    if args.eval_agent:
        # only render videos, do not perform learning
        assert (
            "model_folder" in param_sweeper_dict
        ), "model_folder not found in param_sweeper_dict"
        model_path = args.out_dir + param_sweeper_dict["model_folder"] + str(run_id)
        try:
            train_agent.policy_learner.load_model(path=model_path)

            for i in range(len(param_sweeper_dict["preprocessors"])):
                p = param_sweeper_dict["preprocessors"][i]
                if isinstance(p, ObservationNormalization):
                    with open(
                        model_path + "_observation_norm.pkl",
                        "rb",
                    ) as file:
                        param_sweeper_dict["preprocessors"][i] = pickle.load(file)
        except Exception as e:
            logger.info(f"Failed to load model from {model_path}: {e}")
            exit(0)

        if isinstance(train_agent.policy_learner.exploration_module, Warmup):
            train_agent.policy_learner.exploration_module.warmup_steps = 0
        if isinstance(
            train_agent.policy_learner.exploration_module, EGreedyExploration
        ):
            train_agent.policy_learner.exploration_module.warmup_steps = None
            train_agent.policy_learner.exploration_module.curr_epsilon = (
                train_agent.policy_learner.exploration_module.end_epsilon
            )
        evaluate(
            agent=train_agent,
            env=envs[2] if envs[2] is not None else envs[0],
            render=args.render,
            output_dir=args.out_dir,
            video_dir=args.out_dir + param_sweeper_dict["video_folder"],
            name=training_env_name + "_" + param_sweeper_dict["policy_learner:type"],
            max_steps=param_sweeper_dict["eval_max_steps"],
            run_idx=run_id,
            preprocessors=param_sweeper_dict["preprocessors"],
        )
    else:
        eval_agent = copy.deepcopy(train_agent)
        eval_agent.policy_learner = train_agent.policy_learner
        eval_agent.replay_buffer = train_agent.replay_buffer.__class__(capacity=0)

        if param_sweeper_dict["env"][0]["is_continuing"] or param_sweeper_dict.get(
            "run_steps", False
        ):
            run_steps(train_agent, eval_agent, envs, param_sweeper_dict)
        else:
            run_episodes(train_agent, eval_agent, envs, param_sweeper_dict)

        if param_sweeper_dict["save_model"]:
            assert (
                "model_folder" in param_sweeper_dict
            ), "model_folder not found in param_sweeper_dict"
            model_path = args.out_dir + param_sweeper_dict["model_folder"] + str(run_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            train_agent.policy_learner.save_model(path=model_path)

            for p in param_sweeper_dict["preprocessors"]:
                if isinstance(p, ObservationNormalization):
                    with open(
                        model_path + "_observation_norm.pkl",
                        "wb",
                    ) as file:
                        pickle.dump(p, file)
