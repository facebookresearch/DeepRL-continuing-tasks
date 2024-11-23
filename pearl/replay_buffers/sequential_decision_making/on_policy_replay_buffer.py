# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
from dataclasses import dataclass

from typing import Optional

import numpy as np

import torch

from pearl.api.observation import Observation
from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch


@dataclass(frozen=False)
class OnPolicyTransition(Transition):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_log_probs: Optional[torch.Tensor] = None  # action probs
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward
    value: Optional[torch.Tensor] = None  # value


@dataclass(frozen=False)
class OnPolicyTransitionBatch(TransitionBatch):
    gae: Optional[torch.Tensor] = None  # generalized advantage estimation
    lam_return: Optional[torch.Tensor] = None  # lambda return
    action_log_probs: Optional[torch.Tensor] = None  # action probs
    cum_reward: Optional[torch.Tensor] = None  # cumulative reward
    value: Optional[torch.Tensor] = None  # value

    @classmethod
    def from_parent(
        cls,
        parent_obj: TransitionBatch,
        gae: Optional[torch.Tensor] = None,
        lam_return: Optional[torch.Tensor] = None,
        action_log_probs: Optional[torch.Tensor] = None,
        cum_reward: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> "OnPolicyTransitionBatch":
        # Extract attributes from parent_obj using __dict__ and create a new Child object
        child_obj = cls(
            **parent_obj.__dict__,
            gae=gae,
            lam_return=lam_return,
            action_log_probs=action_log_probs,
            cum_reward=cum_reward,
            value=value,
        )
        return child_obj


class OnPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(
        self,
        capacity: int,
    ) -> None:
        super(OnPolicyReplayBuffer, self).__init__(
            capacity=capacity,
        )
        self._count: int = 0
        self._indices: Optional[np.ndarray] = None

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

    def init_indices(self) -> None:
        """
        Before sampling from the replay buffer,
        first randomly initialize the sequence of indices to be sampled.
        """
        self._count = 0
        self._indices = np.arange(self.pos)
        np.random.shuffle(self._indices)

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        The OnPolicyReplayBuffer modifies the sample method from TensorBasedReplayBuffer.
        Instead of independently drawing random samples each time,
        it shuffles all transitions and samples the first batch_size data.
        On subsequent calls, it samples the next batch_size data until all transitions are sampled once.
        Then, it reshuffles and repeats the process, ensuring a random and even sampling of transitions.
        """
        if self._count == self.pos or self._indices is None:
            self.init_indices()
            self._count = 0
        batch_inds = self._indices[self._count : self._count + batch_size]

        batch = OnPolicyTransitionBatch(
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
            action_log_probs=self.action_log_probs[batch_inds, :],
            gae=self.gae[batch_inds, :],
            lam_return=self.lam_return[batch_inds, :],
            value=self.value[batch_inds, :],
        ).to(self.device_for_batches)
        self._count += batch_size

        return batch

    def sample_all(self) -> TransitionBatch:
        """
        The OnPolicyReplayBuffer modifies the sample method from TensorBasedReplayBuffer.
        Instead of independently drawing random samples each time,
        it shuffles all transitions and samples the first batch_size data.
        On subsequent calls, it samples the next batch_size data until all transitions are sampled once.
        Then, it reshuffles and repeats the process, ensuring a random and even sampling of transitions.
        """

        batch = TransitionBatch(
            state=self.states[: self.pos, :],
            action=self.actions[: self.pos, :],
            reward=self.rewards[: self.pos],
            terminated=self.terminateds[: self.pos],
            truncated=self.truncateds[: self.pos],
            next_state=(
                self.next_states[: self.pos, :]
                if self.next_states is not None
                else None
            ),
        ).to(self.device_for_batches)

        return batch
