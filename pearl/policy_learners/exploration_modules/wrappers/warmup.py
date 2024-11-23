# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional

import torch
import random
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.exploration_modules.exploration_module_wrapper import (
    ExplorationModuleWrapper,
)


class Warmup(ExplorationModuleWrapper):
    """
    Follow the random policy for the first `warmup_steps` steps,
    then switch to the actions from the exploration module.
    """

    def __init__(
        self,
        exploration_module: ExplorationModule,
        warmup_steps: int,
        has_agent_reset: bool = False,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.time_step = 0
        self._test_time = False
        super().__init__(exploration_module)
        self._has_agent_reset = has_agent_reset

    def act(
        self,
        observation: Observation,
        action_space: ActionSpace,
        values: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        if self.time_step < self.warmup_steps:
            action = action_space.sample()
            if self._has_agent_reset:
                reset_prob = 1.0 / random.randint(1, 1000)
                action[-1] = action_space.low[-1] + reset_prob * (
                    action_space.high[-1] - action_space.low[-1]
                )
        else:
            action = self.exploration_module.act(
                observation=observation,
                action_space=action_space,
                values=values,
                exploit_action=exploit_action,
                action_availability_mask=action_availability_mask,
                representation=representation,
            )
        if self._test_time is False:
            self.time_step += 1
        return action

    def set_test_time_false(self) -> None:
        self._test_time = False
        if hasattr(self.exploration_module, "set_test_time_false"):
            self.exploration_module.set_test_time_false()

    def set_test_time_true(self) -> None:
        self._test_time = True
        if hasattr(self.exploration_module, "set_test_time_true"):
            self.exploration_module.set_test_time_true()
