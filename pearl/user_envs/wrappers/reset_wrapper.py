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
        is_continuing: If True, the env is used in a continuing task.
            The env will be reset whenever a termination signal is received.
            But the agent does not care about the reset signal.
            Otherwise, the env is used in an episodic task.
            The step function returns terminated=True and the env should be reset externally.
    """

    def __init__(self, env, reset_cost, random_reset_prob, is_continuing):
        super(ResetWrapper, self).__init__(env)
        assert reset_cost is not None
        assert random_reset_prob is not None
        assert is_continuing is not None
        self.reset_cost = reset_cost
        self.step_cnt = 0
        self.is_continuing = is_continuing
        self.random_reset_prob = random_reset_prob

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_cnt += 1
        resetting = random.random() < self.random_reset_prob
        if resetting or terminated:
            terminated = True
            info["reset"] = True
            reward -= self.reset_cost
            if self.is_continuing:
                obs, _ = self.env.reset(seed=self.step_cnt)
                terminated = False
        else:
            info["reset"] = False
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
        obs, reward, terminated, truncated, info = self.env.step(action[:-1])
        assert terminated is False
        assert truncated is False
        self.step_cnt += 1
        if isinstance(self.action_space, spaces.Box):
            resetting_prob = (action[-1] - self.augmented_low) / (
                self.augmented_high - self.augmented_low
            )
            resetting = random.random() < resetting_prob
            info["reset"] = resetting
            if resetting:
                obs, _ = self.env.reset(seed=self.step_cnt)
                reward -= self.reset_cost
        else:
            raise NotImplementedError("Only Box action space is supported.")

        return obs, reward, terminated, truncated, info
