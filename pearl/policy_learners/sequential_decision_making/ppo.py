# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import math
from typing import Any, Dict, Optional, Union

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    action_scaling,
    ActorNetwork,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.on_policy_replay_buffer import (
    OnPolicyReplayBuffer,
    OnPolicyTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.preprocessing import RunningMeanStd
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC
from torch import nn, optim


class ProximalPolicyOptimization(ActorCriticBase):
    """
    paper: https://arxiv.org/pdf/1707.06347.pdf.
    This class implements both discrete and continuous control versions of PPO.
    """

    def __init__(
        self,
        actor_network_instance: ActorNetwork,
        critic_network_instance: Union[ValueNetwork, nn.Module],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        is_action_continuous: bool,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        epsilon: float = 0.0,
        trace_decay_param: float = 0.95,
        entropy_bonus_scaling: float = 0.0,
        norm_return: bool = False,
        norm_adv: bool = False,
        critic_weight: float = 0.5,
        max_grad_norm: Optional[float] = None,
        anneal_lr: bool = True,
        max_steps: Optional[int] = None,
        clip_value: bool = False,
        reprocessing_buffer: bool = True,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
    ) -> None:
        if exploration_module is None:
            if is_action_continuous:
                exploration_module = NoExploration()
            else:
                exploration_module = PropensityExploration()
        super(ProximalPolicyOptimization, self).__init__(
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            ensemble_critic_size=1,  # not used
            exploration_module=exploration_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=is_action_continuous,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )
        self._epsilon = epsilon
        self._trace_decay_param = trace_decay_param
        self._entropy_bonus_scaling = entropy_bonus_scaling
        self._norm_return = norm_return
        self._norm_adv = norm_adv
        self._critic_weight = critic_weight
        self._max_grad_norm = max_grad_norm
        if self._norm_return:
            self._ret_rms: RunningMeanStd = RunningMeanStd(shape=(1,))
        self._anneal_lr = anneal_lr
        self._max_steps = max_steps
        self._clip_value = clip_value
        self._reprocessing_buffer = reprocessing_buffer

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Loss = actor loss + critic loss + entropy_bonus_scaling * entropy loss
        """
        # TODO: change the output shape of value networks
        assert isinstance(batch, OnPolicyTransitionBatch)
        if self._is_action_continuous:
            action_log_probs, entropy = self._actor.get_action_log_prob_and_entropy(
                state_batch=batch.state,
                action_batch=batch.action,
            )  # shape (batch_size, 1), (batch_size, 1)
        else:
            action_log_probs, entropy = self._actor.get_action_log_prob_and_entropy(
                state_batch=batch.state,
                action_batch=batch.action,
            )  # shape (batch_size, 1), (batch_size, 1)
        # actor loss
        action_log_probs_old = batch.action_log_probs
        r_thelta = torch.exp(
            action_log_probs - action_log_probs_old
        )  # shape (batch_size, 1)
        clip = torch.clamp(
            r_thelta, min=1.0 - self._epsilon, max=1.0 + self._epsilon
        )  # shape (batch_size, 1)
        if self._norm_adv:
            adv = (batch.gae - batch.gae.mean()) / (batch.gae.std() + 1e-8)
            loss = torch.mean(-torch.min(r_thelta * adv, clip * adv))
        else:
            loss = torch.mean(-torch.min(r_thelta * batch.gae, clip * batch.gae))
        loss -= self._entropy_bonus_scaling * torch.mean(entropy)
        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert isinstance(batch, OnPolicyTransitionBatch)
        assert batch.lam_return is not None
        if self._norm_return:  # normalize the return
            assert batch.lam_return is not None
            lam_return = batch.lam_return / math.sqrt(float(self._ret_rms.var) + 1e-8)
        else:
            lam_return = batch.lam_return
        value = self._critic(batch.state)  # shape (batch_size)

        if self._clip_value:
            assert batch.value is not None
            v_clip = batch.value + (value - batch.value).clamp(
                -self._epsilon,
                self._epsilon,
            )
            vf1 = (lam_return - value).pow(2)
            vf2 = (lam_return - v_clip).pow(2)
            return torch.max(vf1, vf2).mean()
        else:
            return (lam_return - value).pow(2).mean()

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        actor_loss = self._actor_loss(batch)
        critic_loss = self._critic_loss(batch)
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        loss = actor_loss + self._critic_weight * critic_loss
        loss.backward()
        if self._max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self._actor.parameters())
                + list(self._critic.parameters()),
                self._max_grad_norm,
            )
        self._actor_optimizer.step()
        self._critic_optimizer.step()
        report = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "reward_rate_estimate": self.reward_rate.item(),
        }
        return report

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        if isinstance(self.reward_centering, RVI_RC):
            freq = self.reward_centering.ref_states_update_freq
            if self._training_steps % freq == 0:
                batch = replay_buffer.create_f_batch(batch_size=self._batch_size, last_k_steps=freq)
                self.reward_centering.f_batch = self.preprocess_batch(batch)
        if self._anneal_lr:
            assert self._max_steps is not None
            frac = 1.0 - (self._current_steps - 1.0) / self._max_steps
            self._actor_optimizer.param_groups[0]["lr"] = (
                frac * self._actor_learning_rate
            )
            self._critic_optimizer.param_groups[0]["lr"] = (
                frac * self._critic_learning_rate
            )
            if isinstance(self.reward_centering, TD_RC):
                self.reward_centering.optimizer.param_groups[0]["lr"] = (
                    frac * self.reward_centering.init_reward_rate_learning_rate
                )
        if len(replay_buffer) == 0:
            return {}

        if self._batch_size == -1 or len(replay_buffer) < self._batch_size:
            batch_size = len(replay_buffer)
        else:
            batch_size = self._batch_size

        report = {}
        for tr in range(self._training_rounds):
            if tr == 0:
                self.preprocess_replay_buffer(
                    replay_buffer, update_action_log_prob=True
                )
            elif (
                self._reprocessing_buffer
                and tr % (len(replay_buffer) // batch_size) == 0
            ):
                self.preprocess_replay_buffer(
                    replay_buffer, update_action_log_prob=False
                )
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

    def preprocess_replay_buffer(
        self, replay_buffer: ReplayBuffer, update_action_log_prob: bool = True
    ) -> None:
        """
        Preprocess the replay buffer by calculating
        and adding the generalized advantage estimates (gae),
        truncated lambda returns (lam_return) and action log probabilities (action_log_probs)
        under the current policy.
        See https://arxiv.org/abs/1707.06347 equation (11) for the definition of gae.
        See "Reinforcement Learning: An Introduction" by Sutton and Barto (2018) equation (12.10)
        for the definition of truncated lambda return.
        """
        if isinstance(self.reward_centering, RVI_RC):
            self.reward_rate = self.compute_f_value(self.reward_centering.f_batch)
        assert type(replay_buffer) is OnPolicyReplayBuffer
        replay_buffer.init_indices()
        batch = replay_buffer.sample_all()
        self.preprocess_batch(batch)
        state_values = self._critic(batch.state).detach().cpu()  # shape (batch_size, 1)
        next_state_values = (
            self._critic(batch.next_state).detach().cpu()
        )  # shape (batch_size, 1)

        if self._norm_return:  # unnormalize state_values
            state_values = state_values * math.sqrt(float(self._ret_rms.var) + 1e-8)
            next_state_values = next_state_values * math.sqrt(
                float(self._ret_rms.var) + 1e-8
            )
        if update_action_log_prob:
            if self._is_action_continuous:
                action_log_probs, _ = self._actor.get_action_log_prob_and_entropy(
                    state_batch=batch.state,
                    action_batch=batch.action,
                )  # shape (batch_size, 1)
            else:
                action_log_probs, _ = self._actor.get_action_log_prob_and_entropy(
                    state_batch=batch.state,
                    action_batch=batch.action,
                )  # shape (batch_size, 1)
            action_log_probs = action_log_probs.detach().cpu()
            replay_buffer.action_log_probs = action_log_probs

        reward = batch.reward.view(-1, 1).cpu()  # shape (batch_size, 1)
        terminated = batch.terminated.view(-1, 1).cpu()  # shape (batch_size, 1)
        truncated = batch.truncated.view(-1, 1).cpu()  # shape (batch_size, 1)
        if isinstance(self.reward_centering, TD_RC):
            if self.reward_centering.initialize_reward_rate == True:
                self.reward_rate.data.fill_(batch.reward.mean())
                self.reward_centering.initialize_reward_rate = False

            self.reward_centering.optimizer.zero_grad()
            td_errors = (
                reward
                - self.reward_rate.cpu()
                + self._discount_factor
                * next_state_values
                * torch.logical_not(terminated)
                - state_values
            )  # shape (batch_size, 1)
            reward_rate_error = td_errors.pow(2).mean()
            reward_rate_error.backward()
            self.reward_centering.optimizer.step()
        td_errors = (
            reward
            - self.reward_rate.detach().cpu()
            + self._discount_factor * next_state_values * torch.logical_not(terminated)
            - state_values
        )  # shape (batch_size, 1)

        discounting = (
            torch.logical_not(torch.logical_or(terminated, truncated))
            * self._discount_factor
            * self._trace_decay_param
        )  # shape (batch_size, 1)

        replay_buffer.gae = torch.zeros_like(td_errors)
        replay_buffer.gae[-1] = td_errors[-1]

        for i in range(replay_buffer.pos - 2, -1, -1):
            replay_buffer.gae[i] = (
                td_errors[i] + discounting[i] * replay_buffer.gae[i + 1]
            )  # shape (1)
        replay_buffer.lam_return = replay_buffer.gae + state_values
        replay_buffer.value = state_values
        if self._norm_return:
            self._ret_rms.update(replay_buffer.lam_return.cpu().numpy())

    def action_post_processing(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Post-process the action before sending to the environment.
        """
        if self._is_action_continuous:
            action = torch.clamp(action, min=-1.0, max=1.0)
            action = action_scaling(self._action_space, action)
        return action

    def compute_f_value(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Computes the f value for a batch of transitions.
        Args:
            batch (TransitionBatch): A batch of transitions.
        Returns:
            f_value (Tensor): The value function for the batch of transitions.
        """
        vs = self._critic(batch.state)
        f_value = torch.mean(vs).detach()
        if self._norm_return:  # unnormalize state_values
            f_value = f_value * math.sqrt(float(self._ret_rms.var) + 1e-8)
        return f_value
