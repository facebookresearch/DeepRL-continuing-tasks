# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")
import random

import numpy as np
from gymnasium import spaces


class ResetWrapper(gym.Wrapper):
    r"""A wrapper that deals with environment-specified resetting and random resetting.
    Args:
        reset_prob: the probability of resetting the env at each step.
        reset_cost: the cost of resetting the env.
    """

    def __init__(self, env, reset_cost, random_reset_prob):
        super(ResetWrapper, self).__init__(env)
        assert reset_cost is not None
        assert random_reset_prob is not None
        self.reset_cost = reset_cost
        self.step_cnt = 0
        self.random_reset_prob = random_reset_prob
        self.reset_transition = False

    def step(self, action):
        if self.reset_transition:
            self.reset_transition = False
            obs, _ = self.env.reset(seed=self.step_cnt)
            self.step_cnt += 1
            terminated = False
            truncated = False
            info = {
                "reset": True,
                "do_not_clip_reset_cost": True,
                "reset_cost": self.reset_cost,
            }
            reward = -self.reset_cost
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_cnt += 1
            resetting = random.random() < self.random_reset_prob
            if resetting or terminated or truncated:
                self.reset_transition = True
                # print(terminated, truncated)
                terminated = False
                truncated = False
        # print(self.step_cnt, self.reset_transition, reward)
        return obs, reward, terminated, truncated, info


# class ResetWrapper(gym.Wrapper):
#     r"""A wrapper that deals with environment-specified resetting and random resetting.
#     Args:
#         reset_prob: the probability of resetting the env at each step.
#         reset_cost: the cost of resetting the env.
#     """

#     def __init__(self, env, reset_cost, random_reset_prob):
#         super(ResetWrapper, self).__init__(env)
#         assert reset_cost is not None
#         assert random_reset_prob is not None
#         self.reset_cost = reset_cost
#         self.step_cnt = 0
#         self.random_reset_prob = random_reset_prob

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         self.step_cnt += 1
#         resetting = random.random() < self.random_reset_prob
#         if resetting or terminated or truncated:
#             obs, _ = self.env.reset(seed=self.step_cnt)
#             terminated = False
#             truncated = False
#             info = {
#                 "reset": True,
#                 "do_not_clip_reset_cost": True,
#                 "reset_cost": self.reset_cost,
#             }
#             reward = -self.reset_cost
#         # print(self.step_cnt, self.reset_transition, reward)
#         return obs, reward, terminated, truncated, info


class EpisodicWrapper(gym.Wrapper):
    r"""A wrapper for episodic tasks
    Args:
        reset_prob: the probability of resetting the env at each step.
        reset_cost: a number added to the final reward.
    """

    def __init__(self, env, reset_cost, random_reset_prob):
        super(EpisodicWrapper, self).__init__(env)
        assert reset_cost is not None
        assert random_reset_prob is not None
        self.reset_cost = reset_cost
        self.random_reset_prob = random_reset_prob

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        resetting = random.random() < self.random_reset_prob
        if resetting:
            terminated = True
        if terminated or truncated:
            reward -= self.reset_cost
            info["do_not_clip_reset_cost"] = True
            info["reset_cost"] = self.reset_cost
        return obs, reward, terminated, truncated, info


class AgentResetWrapper(gym.Wrapper):
    r"""
    This wrapper is used for learning resetting in continuing tasks without environment-specified resetting.
    The agent has one more dim in the action space.
    This additional action chooses whether to reset or not. A cost is incurred if the agent resets.
    Args:
        env: the environment
        reset_cost: the cost of resetting the environment
    """

    def __init__(self, env, reset_cost):
        super(AgentResetWrapper, self).__init__(env)
        assert reset_cost is not None
        self.reset_cost = reset_cost
        self.step_cnt = 0
        if isinstance(self.action_space, spaces.Box):
            # add a dimension corresponding to reset
            self.augmented_low = self.action_space.low[-1]
            self.augmented_high = self.action_space.high[-1]
            tmp = self.action_space.low.tolist()
            tmp.append(self.augmented_low)
            low = np.array(tmp)
            tmp = self.action_space.high.tolist()
            tmp.append(self.augmented_high)
            high = np.array(tmp)
            self.augmented_action_space = spaces.Box(
                low=low,
                high=high,
            )
        else:
            # self.augmented_action_space = spaces.Discrete(n=self.action_space.n + 1)
            raise NotImplementedError("Only Box action space is supported.")

    def step(self, action):
        # tmp = self.env.env.env.env.data.qpos.flat.copy()
        obs, reward, terminated, truncated, info = self.env.step(action[:-1])
        terminated = False
        # assert terminated is False
        assert truncated is False
        self.step_cnt += 1
        if isinstance(self.action_space, spaces.Box):
            # if self.step_cnt > 25000 and self.step_cnt % 100 == 0:
            #     print(action[-1])
            resetting_prob = (action[-1] - self.augmented_low) / (
                self.augmented_high - self.augmented_low
            )
            resetting = random.random() < resetting_prob
            # resetting = obs[0] < 1.0 or obs[0] > 2.0
            if resetting:
                obs, _ = self.env.reset(seed=self.step_cnt)
                reward = -self.reset_cost
                info = {
                    "reset": True,
                    "do_not_clip_reset_cost": True,
                    "reset_cost": self.reset_cost,
                }
            # if not resetting and self.step_cnt > 25000:
            #     if reward < 4:
            #         print("reward", reward, tmp, action[-1])
            # print(self.step_cnt, obs[0], resetting, reward)
        else:
            raise NotImplementedError("Only Box action space is supported.")

        return obs, reward, terminated, truncated, info
