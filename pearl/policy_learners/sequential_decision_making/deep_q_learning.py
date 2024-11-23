# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Optional

import torch
from torch import optim
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.deep_td_learning import (
    DeepTDLearning,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


class DeepQLearning(DeepTDLearning):
    """
    Deep Q Learning Policy Learner
    """

    def __init__(
        self,
        network_instance: QValueNetwork,
        optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        discount_factor: float = 0.99,
        training_rounds: int = 10,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.75,  # a value of 1 indicates no soft updates
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
        **kwargs: Any,
    ) -> None:
        """Constructs a DeepQLearning policy learner. DeepQLearning is based on DeepTDLearning
        class and uses `act` and `learn_batch` methods of that class. We only implement the
        `get_next_state_values` function to compute the bellman targets using Q-learning.

        Args:
            state_dim: Dimension of the observation space.
            action_space (ActionSpace, optional): Action space of the problem. It is kept optional
                to allow for the use of dynamic action spaces (both `learn_batch` and `act`
                functions). Defaults to None.
            hidden_dims (List[int], optional): Hidden dimensions of the default `QValueNetwork`
                (taken to be `VanillaQValueNetwork`). Defaults to None.
            exploration_module (ExplorationModule, optional): Optional exploration module to
                trade-off between exploitation and exploration. Defaults to None.
            learning_rate (float): Learning rate for AdamW optimizer. Defaults to 0.001.
                Note: We use AdamW by default for all value based methods.
            discount_factor (float): Discount factor for TD updates. Defaults to 0.99.
            training_rounds (int): Number of gradient updates per environment step.
                Defaults to 10.
            batch_size (int): Sample size for mini-batch gradient updates. Defaults to 128.
            target_update_freq (int): Frequency at which the target network is updated.
                Defaults to 10.
            soft_update_tau (float): Coefficient for soft updates to the target networks.
                Defaults to 0.01.
            is_conservative (bool): Whether to use conservative updates for offline learning
                with conservative Q-learning (CQL). Defaults to False.
            conservative_alpha (float, optional): Alpha parameter for CQL. Defaults to 2.0.
            network_type (Type[QValueNetwork]): Network type for the Q-value network. Defaults to
                `VanillaQValueNetwork`. This means that by default, an instance of the class
                `VanillaQValueNetwork` (or the specified `network_type` class) is created and used
                for learning.
            action_representation_module (ActionRepresentationModule, optional): Optional module to
                represent actions as a feature vector. Typically specified at the agent level.
                Defaults to None.
            network_instance (QValueNetwork, optional): A network instance to be used as the
                Q-value network. Defaults to None.
                Note: This is an alternative to specifying a `network_type`. If provided, the
                specified `network_type` is ignored and the input `network_instance` is used for
                learning. Allows for custom implementations of Q-value networks.
        """

        super(DeepQLearning, self).__init__(
            exploration_module=exploration_module,
            soft_update_tau=soft_update_tau,
            action_representation_module=action_representation_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            network_instance=network_instance,
            target_update_freq=target_update_freq,
            optimizer=optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
            **kwargs,
        )
        self.all_action_batch = None

    @torch.no_grad()
    def get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        """
        Computes the maximum Q-value over all available actions in the next state using the target
        network. Note: Q-learning is designed to work with discrete action spaces.

        Args:
            batch (TransitionBatch): Batch of transitions.
            batch_size (int): Size of the batch.

        Returns:
            torch.Tensor: Maximum Q-value over all available actions in the next state.
        """

        next_state = batch.next_state  # (batch_size x state_dim)
        assert next_state is not None

        if self.all_action_batch is None or self.all_action_batch.shape[0] != next_state.shape[0]:
            self.all_action_batch = self._action_representation_module(
                self._action_space.actions_batch.unsqueeze(0).repeat(
                    next_state.shape[0], 1, 1
                ).to(self.device)
            )

        # Get Q values for each (state, action), where action \in {available_actions}
        next_state_action_values = self._Q_target.get_q_values(
            state_batch=next_state,
            action_batch=self.all_action_batch,
        )  # (batch_size x action_space_size)

        # Torch.max(1) returns value, indices
        return next_state_action_values.max(1)[0]  # (batch_size)
