# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional

import numpy as np

import torch

from pearl.api.observation import Observation
from pearl.api.reward import Reward
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.device import get_default_device


class TensorBasedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
    ) -> None:
        super(TensorBasedReplayBuffer, self).__init__()
        self.capacity = capacity
        self.observations: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self.terminateds: Optional[np.ndarray] = None
        self.truncateds: Optional[np.ndarray] = None
        self.next_observations: Optional[np.ndarray] = None
        self.pos = 0
        self.full = False
        self._device_for_batches: torch.device = get_default_device()

    def add(
        self,
        obs: Observation,
        action: np.ndarray,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        next_obs: Observation,
    ) -> None:
        if self.capacity == 0:
            return
        if self.observations is None:
            self.observations = np.zeros((self.capacity,) + obs.shape, dtype=obs.dtype)
            # pyre-fixme
            self.actions = np.zeros((self.capacity,) + action.shape, dtype=action.dtype)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.terminateds = np.zeros(self.capacity, dtype=bool)
            self.truncateds = np.zeros(self.capacity, dtype=bool)
            self.next_observations = np.zeros(
                (self.capacity,) + next_obs.shape, dtype=next_obs.dtype
            )
        # pyre-fixme
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminateds[self.pos] = terminated
        self.truncateds[self.pos] = truncated
        self.next_observations[self.pos] = next_obs
        self.pos += 1
        if self.pos == self.capacity:
            self.pos = 0
            self.full = True

    @property
    def device_for_batches(self) -> torch.device:
        return self._device_for_batches

    @device_for_batches.setter
    def device_for_batches(self, new_device_for_batches: torch.device) -> None:
        self._device_for_batches = new_device_for_batches

    def create_f_batch(
        self, batch_size: int, last_k_steps: int = 10000
    ) -> TransitionBatch:
        """
        Create a batch of Transition objects with random state, action, reward,
        next_state, next_action, and terminated.
        """
        assert batch_size <= self.pos
        batch_inds = np.random.randint(
            max(self.pos - last_k_steps, 0), self.pos, size=batch_size
        )
        batch = TransitionBatch(
            # pyre-fixme
            state=self.observations[batch_inds, :],
            action=self.actions[batch_inds, :],
            reward=self.rewards[batch_inds],
            terminated=self.terminateds[batch_inds],
            truncated=self.truncateds[batch_inds],
            next_state=(
                self.next_observations[batch_inds, :]
                if self.next_observations is not None
                else None
            ),
        ).to(self.device_for_batches)
        return batch

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        The shapes of input and output are:
        input: batch_size

        output: TransitionBatch(
          state = tensor(batch_size, state_dim),
          action = tensor(batch_size, action_dim),
          reward = tensor(batch_size, ),
          next_state = tensor(batch_size, state_dim),
          terminated = tensor(batch_size, ),
          truncated = tensor(batch_size, ),
        )
        """
        assert self.capacity > 0
        if batch_size > len(self):
            raise ValueError(
                f"Can't get a batch of size {batch_size} from a replay buffer with"
                f"only {len(self)} elements"
            )
        if self.full is True:
            batch_inds = np.random.randint(0, self.capacity, size=batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)

        batch = TransitionBatch(
            # pyre-fixme
            state=self.observations[batch_inds, :],
            action=self.actions[batch_inds, :],
            reward=self.rewards[batch_inds],
            terminated=self.terminateds[batch_inds],
            truncated=self.truncateds[batch_inds],
            next_state=(
                self.next_observations[batch_inds, :]
                if self.next_observations is not None
                else None
            ),
        ).to(self.device_for_batches)
        return batch

    def __len__(self) -> int:
        if self.full is True:
            return self.capacity
        else:
            return self.pos

    def clear(self) -> None:
        self.observations = None
        self.pos = 0
        self.full = False
