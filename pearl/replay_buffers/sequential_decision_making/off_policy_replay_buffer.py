# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict


from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.observation import Observation
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer


class OffPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int) -> None:
        super(OffPolicyReplayBuffer, self).__init__(
            capacity=capacity,
        )

    # TODO: add helper to convert subjective state into tensors
    # TODO: assumes action space is gym action space with one-hot encoding
    def push(
        self,
        state: Observation,
        action: Action,
        reward: Reward,
        next_state: Observation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
        )
