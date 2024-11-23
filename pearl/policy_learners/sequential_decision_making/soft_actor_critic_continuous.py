# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, Optional, Union

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    ensemble_critic_action_value_loss,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from torch import nn, optim


class ContinuousSoftActorCritic(ActorCriticBase):
    """
    Soft Actor Critic Policy Learner.
    """

    def __init__(
        self,
        actor_network_instance: ActorNetwork,
        critic_network_instance: Union[QValueNetwork, nn.Module],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 256,
        entropy_coef: float = 0.2,
        entropy_autotune: bool = True,
        ensemble_critic_size: int = 2,
        target_entropy_offset: float = 0.0,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
    ) -> None:
        super(ContinuousSoftActorCritic, self).__init__(
            use_actor_target=False,
            use_critic_target=True,
            actor_soft_update_tau=0.0,
            critic_soft_update_tau=critic_soft_update_tau,
            ensemble_critic_size=ensemble_critic_size,
            exploration_module=exploration_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=True,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )

        self._entropy_autotune = entropy_autotune
        if entropy_autotune:
            # initialize the entropy coefficient to 0
            self.register_parameter(
                "_log_entropy",
                torch.nn.Parameter(torch.zeros(1, requires_grad=True)),
            )
            self._entropy_optimizer: torch.optim.Optimizer = optim.Adam(
                [self._log_entropy], lr=self._critic_learning_rate
            )
            self.register_buffer("_entropy_coef", torch.exp(self._log_entropy).detach())
            self.register_buffer(
                "_target_entropy",
                -torch.tensor(action_representation_module.representation_dim) + target_entropy_offset,
            )
        else:
            self.register_buffer("_entropy_coef", torch.tensor(entropy_coef))

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        actor_critic_loss = super().learn_batch(batch)

        if self._entropy_autotune:

            entropy_optimizer_loss = (
                -torch.exp(self._log_entropy)
                * (self._action_batch_log_prob_cache + self._target_entropy).detach()
            ).mean()

            self._entropy_optimizer.zero_grad()
            entropy_optimizer_loss.backward()
            self._entropy_optimizer.step()

            self._entropy_coef = torch.exp(self._log_entropy).detach()
            {**actor_critic_loss, **{"entropy_coef": entropy_optimizer_loss}}

        return actor_critic_loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:

        reward_batch = batch.reward  # shape: (batch_size)
        terminated_batch = batch.terminated  # shape: (batch_size)

        if terminated_batch is not None:
            expected_state_action_values = (
                self._get_next_state_expected_values(batch)
                * self._discount_factor
                * (1 - terminated_batch.float())
            ) + reward_batch  # shape of expected_state_action_values: (batch_size)
        else:
            raise AssertionError("terminated_batch should not be None")

        loss = ensemble_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            # pyre-fixme
            critic=self._critic,
            reward_rate=self.reward_rate,
        )

        return loss

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        next_state_batch = batch.next_state  # shape: (batch_size x state_dim)

        # shape of next_action_batch: (batch_size, action_dim)
        # shape of next_action_log_prob: (batch_size, 1)
        (
            next_action_batch,
            next_action_batch_log_prob,
        ) = self._actor.sample_action(next_state_batch, get_log_prob=True)

        next_qs = self._critic_target.get_q_values(
            state_batch=next_state_batch,
            action_batch=next_action_batch,
            get_all_values=True,
        )  # shape: (ensemble_critic_size, batch_size)

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.min(next_qs, dim=0).values  # shape: (batch_size)
        next_state_action_values = next_q.unsqueeze(-1)  # shape: (batch_size x 1)

        # add entropy regularization
        next_state_action_values = next_state_action_values - (
            self._entropy_coef * next_action_batch_log_prob
        )
        # shape: (batch_size x 1)

        return next_state_action_values.view(-1)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state  # shape: (batch_size x state_dim)

        # shape of action_batch: (batch_size, action_dim)
        # shape of action_batch_log_prob: (batch_size, 1)
        (
            action_batch,
            action_batch_log_prob,
        ) = self._actor.sample_action(state_batch, get_log_prob=True)
        self._action_batch_log_prob_cache = action_batch_log_prob
        qs = self._critic.get_q_values(
            state_batch=state_batch,
            action_batch=action_batch,
            get_all_values=True,
        )  # shape: (ensemble_critic_size, batch_size)

        # clipped double q learning (reduce overestimation bias)
        q = torch.min(qs, dim=0).values  # shape: (batch_size)
        state_action_values = q.unsqueeze(-1)  # shape: (batch_size x 1)

        loss = (self._entropy_coef * action_batch_log_prob - state_action_values).mean()

        return loss
