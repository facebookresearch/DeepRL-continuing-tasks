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
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    EnsembleQValueNetwork,
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
from torch import nn, optim


# Currently available actions is not used. Needs to be updated once we know the input
# structure of production stack on this param.


# TODO: to make things easier with a single optimizer, we need to polish this method.
class SoftActorCritic(ActorCriticBase):
    """
    Implementation of Soft Actor Critic Policy Learner for discrete action spaces.
    """

    def __init__(
        self,
        actor_network_instance: ActorNetwork,
        critic_network_instance: Union[QValueNetwork, nn.Module],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        critic_soft_update_tau: float = 1,
        critic_target_update_freq: int = 8000,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        entropy_coef: float = 0.2,
        entropy_autotune: bool = True,
        ensemble_critic_size: int = 2,
        target_entropy_scale: float = 0.89,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
    ) -> None:
        super(SoftActorCritic, self).__init__(
            use_actor_target=False,
            use_critic_target=True,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=critic_soft_update_tau,
            actor_target_update_freq=1,
            critic_target_update_freq=critic_target_update_freq,
            ensemble_critic_size=ensemble_critic_size,
            exploration_module=exploration_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=False,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )

        # This is needed to avoid actor softmax overflow issue.
        # Should not be left for users to choose.
        # self.scheduler = optim.lr_scheduler.ExponentialLR(
        #     self._actor_optimizer, gamma=0.99
        # )

        # TODO: implement learnable entropy coefficient
        self._entropy_autotune = entropy_autotune
        if entropy_autotune:
            # initialize the entropy coefficient to 0
            self.register_parameter(
                "_log_entropy",
                torch.nn.Parameter(torch.zeros(1, requires_grad=True)),
            )
            self._entropy_optimizer: torch.optim.Optimizer = optim.Adam(
                [self._log_entropy], lr=self._critic_learning_rate, eps=1e-4
            )
            self.register_buffer("_entropy_coef", torch.exp(self._log_entropy).detach())
            print(action_representation_module.max_number_actions)
            self.register_buffer(
                "_target_entropy",
                -target_entropy_scale * torch.log(1.0 / torch.tensor(action_representation_module.max_number_actions)),
            )
        else:
            self.register_buffer("_entropy_coef", torch.tensor(entropy_coef))
        self.all_action_batch: Optional[torch.Tensor] = None

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        actor_critic_loss = super().learn_batch(batch)

        if self._entropy_autotune:
            entropy = (
                -(self._action_probs_cache * self._action_log_probs_cache).sum(1).mean()
            )
            entropy_optimizer_loss = (
                torch.exp(self._log_entropy) * (entropy - self._target_entropy).detach()
            )

            self._entropy_optimizer.zero_grad()
            entropy_optimizer_loss.backward()
            self._entropy_optimizer.step()

            self._entropy_coef = torch.exp(self._log_entropy).detach()
            {**actor_critic_loss, **{"entropy_coef": entropy_optimizer_loss}}

        return actor_critic_loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:

        reward_batch = batch.reward  # (batch_size)
        terminated_batch = batch.terminated  # (batch_size)

        assert terminated_batch is not None
        expected_state_action_values = (
            self._get_next_state_expected_values(batch)
            * self._discount_factor
            * (1 - terminated_batch.float())
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        assert isinstance(self._critic, EnsembleQValueNetwork)
        loss = ensemble_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
            reward_rate=self.reward_rate,
        )

        return loss

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)

        assert next_state_batch is not None
        # get q values of (states, all actions) from twin critics
        next_qs = self._critic_target.get_q_values(
            state_batch=next_state_batch,
            action_batch=self.all_action_batch,
            get_all_values=True,
        )  # (ensemble_critic_size, batch_size, action_space_size)

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.min(next_qs, dim=0).values  # (batch_size, action_space_size)
        # random ensemble distillation (reduce overestimation bias)
        # random_index = torch.randint(0, 2, (1,)).item()
        # next_q = next_q1 if random_index == 0 else next_q2

        # Make sure that unavailable actions' Q values are assigned to 0.0
        # since we are calculating expectation

        next_state_policy_dist = self._actor.get_policy_distribution(
            state_batch=next_state_batch,
        )  # (batch_size x action_space_size)

        # Entropy Regularization
        next_q = (
            next_q - self._entropy_coef * torch.log(next_state_policy_dist + 1e-8)
        ) * next_state_policy_dist  # (batch_size x action_space_size)

        return next_q.sum(dim=1)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state  # (batch_size x state_dim)
        if self.all_action_batch is None or self.all_action_batch.shape[0] != state_batch.shape[0]:
            self.all_action_batch = self._action_representation_module(
                self._action_space.actions_batch.unsqueeze(0).repeat(
                    state_batch.shape[0], 1, 1
                ).to(self.device)
            )
        # get q values of (states, all actions) from twin critics
        qs = self._critic.get_q_values(
            state_batch=state_batch,
            action_batch=self.all_action_batch,
            get_all_values=True,
        )  # (ensemble_critic_size, batch_size, action_space_size)
        # clipped double q learning (reduce overestimation bias)
        q = torch.min(qs, dim=0).values  # (batch_size, action_space_size)
        new_policy_dist = self._actor.get_policy_distribution(
            state_batch=state_batch,
        )  # (batch_size x action_space_size)
        self._action_probs_cache = new_policy_dist
        self._action_log_probs_cache = torch.log(new_policy_dist + 1e-8)
        
        loss = (
            new_policy_dist * (self._entropy_coef * self._action_log_probs_cache - q)
        ).mean()

        return loss
