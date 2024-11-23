# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod

import torch
from pearl.api.action import Action
from pearl.api.observation import Observation
from pearl.api.reward import Reward


class ReplayBuffer(ABC):
    """
    A base class for all replay buffers.

    Replay buffers store transitions collected from an agent's experience,
    and batches of those transitions can be sampled to train the agent.

    They are stored in the CPU since they may grow quite large,
    but contain a property `device` which specifies where
    batches are stored.
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_action_continuous: bool = False

    @property
    @abstractmethod
    def device_for_batches(self) -> torch.device:
        """
        The device on which _batches_ are stored
        (the replay buffer is always stored in the CPU).
        """
        pass

    @device_for_batches.setter
    @abstractmethod
    def device_for_batches(self, new_device_for_batches: torch.device) -> None:
        pass

    @abstractmethod
    def push(
        self,
        observation: Observation,
        action: Action,
        reward: Reward,
        next_state: Observation,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Saves a transition."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> object:
        pass

    @abstractmethod
    def clear(self) -> None:
        """Empties replay buffer"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def is_action_continuous(self) -> bool:
        """Whether the action space is continuous or not."""
        return self._is_action_continuous

    @is_action_continuous.setter
    def is_action_continuous(self, value: bool) -> None:
        """Set whether the action space is continuous or not."""
        self._is_action_continuous = value