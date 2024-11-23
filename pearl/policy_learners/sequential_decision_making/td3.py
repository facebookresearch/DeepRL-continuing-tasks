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
from pearl.neural_networks.common.utils import update_target_network
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
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    ensemble_critic_action_value_loss,
)
from torch import nn, optim


class TD3(DeepDeterministicPolicyGradient):
    """
    TD3 uses a deterministic actor, Twin critics, and a delayed actor update.
        - An exploration module is used with deterministic actors.
        - To avoid exploration, use NoExploration module.
    """

    def __init__(
        self,
        actor_network_instance: ActorNetwork,
        critic_network_instance: Union[QValueNetwork, nn.Module],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float|torch.Tensor = 0.2,
        actor_update_noise_clip: float = 0.5,
        ensemble_critic_size: int = 2,  # for twin critic, default choice is 2
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
    ) -> None:
        super(TD3, self).__init__(
            exploration_module=exploration_module,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            action_representation_module=action_representation_module,
            ensemble_critic_size=ensemble_critic_size,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )
        self._actor_update_freq = actor_update_freq
        self._actor_update_noise = actor_update_noise
        self._actor_update_noise_clip = actor_update_noise_clip

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # The actor and the critic updates are arranged in the following way
        # for the same reason as in the comment "If the history summarization module ..."
        # in the learn_batch function in actor_critic_base.py.
        if isinstance(self.reward_centering, TD_RC) and self.reward_centering.initialize_reward_rate == True:
            self.reward_rate.data.fill_(batch.reward.mean())
            self.reward_centering.initialize_reward_rate = False
        report = {}
        # delayed actor update
        if self._training_steps % self._actor_update_freq == 0:
            self._actor_optimizer.zero_grad()
            actor_loss = self._actor_loss(batch)
            actor_loss.backward(retain_graph=True)
            self._actor_optimizer.step()
            report["actor_loss"] = actor_loss.item()

        if isinstance(self.reward_centering, TD_RC):
            self.reward_centering.optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        critic_loss = self._critic_loss(batch)  # critic update
        critic_loss.backward()
        self._critic_optimizer.step()
        if isinstance(self.reward_centering, TD_RC):
            self.reward_centering.optimizer.step()
        report["critic_loss"] = critic_loss.item()
        report["reward_rate_estimate"] = self.reward_rate.item()

        if self._training_steps % self._actor_update_freq == 0:
            # update targets of critics using soft updates
            update_target_network(
                self._critic_target,
                self._critic,
                self._critic_soft_update_tau,
            )
            # update target of actor network using soft updates
            update_target_network(
                self._actor_target, self._actor, self._actor_soft_update_tau
            )

        return report

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            next_action = self._actor_target.sample_action(batch.next_state)

            # sample clipped gaussian noise
            if isinstance(self._actor_update_noise, torch.Tensor):
                self._actor_update_noise = self._actor_update_noise.to(batch.state.device)
                tmp = self._actor_update_noise.repeat(next_action.size(0), 1)
                assert tmp.size() == next_action.size()
            noise = torch.normal(
                mean=torch.zeros(next_action.size(), device=batch.device),
                std=tmp if isinstance(self._actor_update_noise, torch.Tensor) else torch.ones(next_action.size(), device=batch.device) * self._actor_update_noise,
            )

            noise = torch.clamp(
                noise,
                -self._actor_update_noise_clip,
                self._actor_update_noise_clip,
            )  # shape (batch_size, action_dim)

            # rescale the noise
            low = self._action_space.low.clone().to(batch.device)
            high = self._action_space.high.clone().to(batch.device)
            noise = noise * (high - low) / 2

            # add clipped noise to next_action
            next_action = torch.clamp(
                next_action + noise, low, high
            )  # shape (batch_size, action_dim)

            # sample q values of (next_state, next_action) from targets of critics
            next_qs = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
                get_all_values=True,
            )  # shape (ensemble_critic_size, batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.min(next_qs, dim=0).values  # shape (batch_size)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.terminated.float())
            ) + batch.reward  # (batch_size)

        # update ensemble critics towards bellman target
        assert isinstance(self._critic, EnsembleQValueNetwork)
        loss = ensemble_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
            reward_rate=self.reward_rate,
        )
        return loss
