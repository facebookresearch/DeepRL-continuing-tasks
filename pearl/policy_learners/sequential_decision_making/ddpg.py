# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional, Union

import torch
from torch import optim
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (  # noqa E501
    NormalDistributionExploration,
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
from torch import nn, optim


class DeepDeterministicPolicyGradient(ActorCriticBase):
    """
    A Class for Deep Deterministic Deep Policy Gradient policy learner.
    paper: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(
        self,
        actor_network_instance: ActorNetwork,
        critic_network_instance: Union[QValueNetwork, nn.Module],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        ensemble_critic_size: int = 1,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
    ) -> None:
        super(DeepDeterministicPolicyGradient, self).__init__(
            use_actor_target=True,
            use_critic_target=True,
            ensemble_critic_size=ensemble_critic_size,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
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

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:

        # sample a batch of actions from the actor network; shape (batch_size, action_dim)
        action_batch = self._actor.sample_action(batch.state)

        # obtain q values for (batch.state, action_batch) from the first critic
        q = self._critic.get_q_values(
            state_batch=batch.state,
            action_batch=action_batch,
            z=0,
        )

        # optimization objective: optimize actor to maximize Q(s, a)
        loss = -q.mean()

        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:

        with torch.no_grad():
            # sample a batch of next actions from target actor network;
            next_action = self._actor_target.sample_action(batch.next_state)
            # (batch_size, action_dim)
            # get q values of (batch.next_state, next_action) from targets of ensemble critic
            next_qs = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
                get_all_values=True,
            )  # shape (ensemble_critic_size, batch_size)

            # clipped double q learning (reduce overestimation bias); shape (batch_size)
            next_q = torch.min(next_qs, dim=0).values  # shape (batch_size)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.terminated.float())
            ) + batch.reward  # shape (batch_size)

        # update ensemble critics towards bellman target
        loss = ensemble_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
            reward_rate=self.reward_rate,
        )

        return loss
