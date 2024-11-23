# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium as gym
import numpy as np


class PusherWrapper(gym.Wrapper):
    r"""wrapper for reacher. We reset the target position every 50 steps."""

    def __init__(self, env):
        super(PusherWrapper, self).__init__(env)
        self.step_cnt = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_cnt += 1
        if self.step_cnt % 100 == 0:  # reset the target position.
            while True:
                self.env.unwrapped.cylinder_pos = np.concatenate(
                    [
                        self.env.unwrapped.np_random.uniform(low=-0.3, high=0, size=1),
                        self.env.unwrapped.np_random.uniform(
                            low=-0.2, high=0.2, size=1
                        ),
                    ]
                )
                if (
                    np.linalg.norm(
                        self.env.unwrapped.cylinder_pos - self.env.unwrapped.goal_pos
                    )
                    > 0.17
                ):
                    break
            self.env.unwrapped.data.qpos[-4:-2] = self.env.unwrapped.cylinder_pos
            self.env.unwrapped.set_state(
                self.env.unwrapped.data.qpos, self.env.unwrapped.data.qvel
            )
            obs = self.env.unwrapped._get_obs()
        return obs, reward, terminated, truncated, info
