# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.observation import Observation
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch


class PolicyLearner(torch.nn.Module, ABC):
    """
    An abstract interface for policy learners.

    Important requirements for policy learners using tensors:
        1. If a policy learner is to operate on a given torch device,
           the policy learner must be moved to that device using method `to(device)`.
        2. All inputs to policy leaners must be moved to the proper device,
           including `TransitionBatch`es (which also have a `to(device)` method).
    """

    # See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use  # noqa E501
    # of `T` to annotate `self`. At least one method of `PolicyLearner`
    # returns `self` and we want those return values to be
    # the type of the subclass, not the looser type of `PolicyLearner`.
    T = TypeVar("T", bound="PolicyLearner")

    def __init__(
        self,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        is_action_continuous: bool,
        training_rounds: int = 100,
        batch_size: int = 1,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
        **options: Any,
    ) -> None:
        super(PolicyLearner, self).__init__()

        self._exploration_module: ExplorationModule = exploration_module
        assert action_representation_module.representation_dim != -1
        self._action_representation_module = action_representation_module

        self._training_rounds = training_rounds
        self._batch_size = batch_size
        self._training_steps = 0
        self._is_action_continuous = is_action_continuous
        self.reward_rate = reward_rate
        self.reward_centering = reward_centering
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def exploration_module(self) -> ExplorationModule:
        return self._exploration_module

    @property
    def action_representation_module(self) -> ActionRepresentationModule:
        return self._action_representation_module

    @exploration_module.setter
    def exploration_module(self, new_exploration_module: ExplorationModule) -> None:
        self._exploration_module = new_exploration_module

    def get_action_representation_module(self) -> ActionRepresentationModule:
        return self._action_representation_module

    def reset(self, action_space: ActionSpace) -> None:
        """Resets policy maker for a new episode. Default implementation does nothing."""
        pass

    @abstractmethod
    def act(
        self,
        observation: Observation,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        pass

    def learn(
        self,
        replay_buffer: ReplayBuffer,
    ) -> Dict[str, Any]:
        """
        Args:
            replay_buffer: buffer instance which learn is reading from

        Returns:
            A dictionary which includes useful metrics
        """
        if len(replay_buffer) == 0:
            return {}

        if self._batch_size == -1 or len(replay_buffer) < self._batch_size:
            batch_size = len(replay_buffer)
        else:
            batch_size = self._batch_size
        if isinstance(self.reward_centering, RVI_RC):
            freq = self.reward_centering.ref_states_update_freq
            if self._training_steps % freq == 0:
                batch = replay_buffer.create_f_batch(batch_size=self._batch_size, last_k_steps=freq)
                self.reward_centering.f_batch = self.preprocess_batch(batch)
            self.reward_rate = self.compute_f_value(self.reward_centering.f_batch)
        report = {}
        for _ in range(self._training_rounds):
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            single_report = {}
            if isinstance(batch, TransitionBatch):
                batch = self.preprocess_batch(batch)
                single_report = self.learn_batch(batch)

            for k, v in single_report.items():
                if k in report:
                    report[k].append(v)
                else:
                    report[k] = [v]
        return report

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """
        Processes a batch of transitions before passing it to learn_batch().
        This function can be used to implement preprocessing steps such as
        transform the actions.
        """

        batch.action = self._action_representation_module(batch.action)
        return batch

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Args:
            batch: batch of data that agent is learning from

        Returns:
            A dictionary which includes useful metrics
        """
        raise NotImplementedError("learn_batch is not implemented")

    def __str__(self) -> str:
        return self.__class__.__name__
