# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random
from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import numpy as np

import torch

from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.observation import Observation
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
        self.states = None
        self.pos = 0
        self.full = False
        self._device_for_batches: torch.device = get_default_device()

    def add(
        self,
        state,
        action,
        reward,
        terminated,
        truncated,
        next_state,
    ) -> None:
        if self.capacity == 0:
            return
        if self.states is None:
            self.states = np.zeros((self.capacity,) + state.shape, dtype=state.dtype)
            self.actions = np.zeros((self.capacity,) + action.shape, dtype=action.dtype)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.terminateds = np.zeros(self.capacity, dtype=bool)
            self.truncateds = np.zeros(self.capacity, dtype=bool)
            self.next_states = np.zeros(
                (self.capacity,) + next_state.shape, dtype=next_state.dtype
            )

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminateds[self.pos] = terminated
        self.truncateds[self.pos] = truncated
        self.next_states[self.pos] = next_state

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

    def _process_single_state(self, state: Observation) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.clone().detach()
        else:
            return torch.tensor(state)

    def _process_single_action(self, action: Action) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            return action.clone().detach()
        else:
            return torch.tensor(action)

    def _process_single_reward(self, reward: Reward) -> torch.Tensor:
        return torch.tensor(reward)

    def _process_single_terminated(self, terminated: bool) -> torch.Tensor:
        return torch.tensor(terminated)  # (1,)

    def _process_single_truncated(self, truncated: bool) -> torch.Tensor:
        return torch.tensor(truncated)  # (1,)

    def create_f_batch(self, batch_size: int, last_k_steps: int = 10000) -> TransitionBatch:
        """
        Create a batch of Transition objects with random state, action, reward,
        next_state, next_action, and terminated.
        """
        assert batch_size <= self.pos
        batch_inds = np.random.randint(
            max(self.pos - last_k_steps, 0), self.pos, size=batch_size
        )
        batch = TransitionBatch(
            state=self.states[batch_inds, :],
            action=self.actions[batch_inds, :],
            reward=self.rewards[batch_inds],
            terminated=self.terminateds[batch_inds],
            truncated=self.truncateds[batch_inds],
            next_state=(
                self.next_states[batch_inds, :]
                if self.next_states is not None
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
        # print(self.states)
        # print(self.actions[batch_inds, :])
        batch = TransitionBatch(
            state=self.states[batch_inds, :],
            action=self.actions[batch_inds, :],
            reward=self.rewards[batch_inds],
            terminated=self.terminateds[batch_inds],
            truncated=self.truncateds[batch_inds],
            next_state=(
                self.next_states[batch_inds, :]
                if self.next_states is not None
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
        self.states = None
        self.pos = 0
        self.full = False
