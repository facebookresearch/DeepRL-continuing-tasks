# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium as gym
import numpy as np


class SwimmerWrapper(gym.Wrapper):
    r"""wrapper for swimmer. We convert the observed rads so that they are within the range of (-\pi, \pi]."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[0] = obs[0] % (2 * np.pi) - np.pi
        obs[1] = obs[1] % (2 * np.pi) - np.pi
        obs[2] = obs[2] % (2 * np.pi) - np.pi
        return obs, reward, terminated, truncated, info
