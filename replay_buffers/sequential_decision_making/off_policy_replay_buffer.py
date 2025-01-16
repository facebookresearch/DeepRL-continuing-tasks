# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import numpy as np

from pearl.api.observation import Observation
from pearl.api.reward import Reward
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer


class OffPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int) -> None:
        super(OffPolicyReplayBuffer, self).__init__(
            capacity=capacity,
        )

    def push(
        self,
        obs: Observation,
        action: np.ndarray,
        reward: Reward,
        next_obs: Observation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )
