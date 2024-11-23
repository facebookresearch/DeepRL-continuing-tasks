# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium as gym


class HalfCheetahWrapper(gym.Wrapper):
    r"""wrapper for half cheetah. Half cheetah may flip its body. We reset it to an initial state when this happens."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (
            self.env.unwrapped.data.body("torso").xpos[2] < 0.15
        ):  # the z axis of the half cheetah body is too low.
            terminated = True
        return obs, reward, terminated, truncated, info
