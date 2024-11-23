# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random
from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation
from pearl.policy_learners.exploration_modules.common.uniform_exploration_base import (
    UniformExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class EGreedyExploration(UniformExplorationBase):
    """
    epsilon Greedy exploration module.
    """

    def __init__(
        self,
        epsilon: float,
        start_epsilon: Optional[float] = None,
        end_epsilon: Optional[float] = None,
        warmup_steps: Optional[int] = None,
    ) -> None:
        super(EGreedyExploration, self).__init__()
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.warmup_steps = warmup_steps
        self.time_step = 0
        self._test_time = False
        if (
            self.start_epsilon is not None
            and self.end_epsilon is not None
            and self.warmup_steps is not None
        ):
            self.curr_epsilon: float = self.start_epsilon
        else:
            self.curr_epsilon = epsilon

    def act(
        self,
        observation: Observation,
        action_space: ActionSpace,
        exploit_action: Optional[Action],
        values: Optional[torch.Tensor] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        if (
            self.start_epsilon is not None
            and self.end_epsilon is not None
            and self.warmup_steps is not None
        ):
            if self.time_step <= self.warmup_steps:
                self.curr_epsilon = (
                    self.start_epsilon
                    + (self.end_epsilon - self.start_epsilon)
                    * self.time_step
                    / self.warmup_steps
                )
            if self._test_time is False:
                self.time_step += 1
        if exploit_action is None:
            raise ValueError(
                "exploit_action cannot be None for epsilon-greedy exploration"
            )
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("action space must be discrete")
        if random.random() < self.curr_epsilon:
            return action_space.sample(action_availability_mask).to(
                exploit_action.device
            )
        else:
            return exploit_action

    def set_test_time_false(self) -> None:
        self._test_time = False

    def set_test_time_true(self) -> None:
        self._test_time = True
