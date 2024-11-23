# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import torch

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation
from pearl.neural_networks.common.utils import (
    update_target_network,
)
from pearl.neural_networks.common.value_networks import ValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import nn, optim


class ActorCriticBase(PolicyLearner):
    """
    A base class for all actor-critic based policy learners.

    Many components that are common to all actor-critic methods have been put in this base class.
    These include:

    - actor and critic network initializations (optionally with corresponding target networks).
    - `act`, `reset` and `learn_batch` methods.
    - Utility functions used by many actor-critic methods.
    """

    def __init__(
        self,
        actor_network_instance: ActorNetwork,
        critic_network_instance: Union[ValueNetwork, QValueNetwork, nn.Module],
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        action_representation_module: ActionRepresentationModule,
        exploration_module: ExplorationModule,
        use_actor_target: bool = False,
        use_critic_target: bool = False,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        actor_target_update_freq: int = 1,
        critic_target_update_freq: int = 1,
        ensemble_critic_size: int = 1,  # number of critics, used only for EnsembleQValueNetwork
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        is_action_continuous: bool = False,
        reward_rate: torch.Tensor = torch.tensor(0.0),
        reward_centering: Optional[TD_RC|RVI_RC|MA_RC] = None,
    ) -> None:
        super(ActorCriticBase, self).__init__(
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
            reward_rate=reward_rate,
            reward_centering=reward_centering,
        )
        """
        Constructs a base actor-critic policy learner.
        """

        self._use_actor_target = use_actor_target
        self._use_critic_target = use_critic_target

        self._actor: nn.Module = actor_network_instance
        self._actor_optimizer: optim.Optimizer = actor_optimizer
        self._actor_target_update_freq = actor_target_update_freq
        self._actor_soft_update_tau = actor_soft_update_tau

        # make a copy of the actor network to be used as the actor target network
        if self._use_actor_target:
            self._actor_target: nn.Module = copy.deepcopy(self._actor)

        self._critic_target_update_freq = critic_target_update_freq
        self._critic_soft_update_tau = critic_soft_update_tau
        self._critic: nn.Module = critic_network_instance
        self._critic_optimizer: optim.Optimizer = critic_optimizer
        if self._use_critic_target:
            self._critic_target: nn.Module = copy.deepcopy(self._critic)

        self._discount_factor = discount_factor
        self._actor_learning_rate = self._actor_optimizer.param_groups[0]["lr"]
        self._critic_learning_rate = self._critic_optimizer.param_groups[0]["lr"]
        self._current_steps = 0
        self._test_time = False

    def act(
        self,
        observation: Observation,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        """
        Determines an action based on the policy network and optionally the exploration module.
        This function can operate in two modes: exploit or explore. The mode is determined by the
        `exploit` parameter.

        - If `exploit` is True, the function returns an action determined solely by the policy
        network.
        - If `exploit` is False, the function first calculates an `exploit_action` using the policy
        network. This action is then passed to the exploration module, along with additional
        arguments specific to the exploration module in use. The exploration module then generates
        an action that strikes a balance between exploration and exploitation.

        Args:
            available_action_space (ActionSpace): Set of eligible actions.
            exploit (bool, optional): Determines the mode of operation. If True, the function
            operates in exploit mode. If False, it operates in explore mode. Defaults to False.
        Returns:
            Action: An action (decision made by the agent in the given subjective state)
            that balances between exploration and exploitation, depending on the mode
            specified by the user. The returned action is from the available action space.
        """
        # Step 1: compute exploit_action
        # (action computed by actor network; and without any exploration)
        if self._test_time is False:
            self._current_steps += 1
        with torch.no_grad():
            if self._is_action_continuous:
                exploit_action = self._actor.sample_action(observation)
                action_probabilities = None
            else:
                assert isinstance(available_action_space, DiscreteActionSpace)
                action_probabilities = self._actor.get_policy_distribution(
                    state_batch=observation,
                )
                # (action_space_size)
                exploit_action_index = torch.argmax(action_probabilities)
                exploit_action = available_action_space.actions[exploit_action_index]

        # Step 2: return exploit action if no exploration,
        # else pass through the exploration module
        if exploit:
            return exploit_action

        # TODO: carefully check if safe action space is integrated with the exploration module
        return self._exploration_module.act(
            exploit_action=exploit_action,
            action_space=available_action_space,
            observation=observation,
            values=action_probabilities,
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Trains the actor and critic networks using a batch of transitions.
        This method performs the following steps:

        1. Updates the actor network with the input batch of transitions.
        2. Updates the critic network with the input batch of transitions.
        3. If using target network for critics (i.e. `use_critic_target` argument is True), the
        function updates the critic target network.
        4. If using target network for policy (i.e. `use_actor_target` argument is True), the
        function updates the actor target network.

        Note: While this method provides a general approach to actor-critic methods, specific
        algorithms may override it to introduce unique behaviors. For instance, the TD3 algorithm
        updates the actor network less frequently than the critic network.

        Args:
            batch (TransitionBatch): Batch of transitions to use for actor and critic updates.
        Returns:
            Dict[str, Any]: A dictionary containing the loss reports from the critic
            and actor updates. These can be useful to track for debugging purposes.
        """
        if isinstance(self.reward_centering, TD_RC):
            if self.reward_centering.initialize_reward_rate == True:
                self.reward_rate.data.fill_(batch.reward.mean())
                self.reward_centering.initialize_reward_rate = False
        actor_loss = self._actor_loss(batch)
        self._actor_optimizer.zero_grad()
        """
        If the history summarization module is a neural network,
        the computation graph of this neural network is used
        to obtain both actor and critic losses.
        Without retain_graph=True, after actor_loss.backward(), the computation graph is cleared.
        After the graph is cleared, critic_loss.backward() fails.
        """
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()
        report = {"actor_loss": actor_loss.item()}
        if isinstance(self.reward_centering, TD_RC):
            self.reward_centering.optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        critic_loss = self._critic_loss(batch)
        critic_loss.backward()
        self._critic_optimizer.step()
        if isinstance(self.reward_centering, TD_RC):
            self.reward_centering.optimizer.step()
        report["critic_loss"] = critic_loss.item()
        report["reward_rate_estimate"] = self.reward_rate.item()

        if (
            self._use_critic_target
            and self._training_steps % self._critic_target_update_freq == 0
        ):
            update_target_network(
                self._critic_target,
                self._critic,
                self._critic_soft_update_tau,
            )
        if (
            self._use_actor_target
            and self._training_steps % self._actor_target_update_freq == 0
        ):
            update_target_network(
                self._actor_target,
                self._actor,
                self._actor_soft_update_tau,
            )
        return report

    @abstractmethod
    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the actor
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the actor network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The actor loss.
        """
        pass

    @abstractmethod
    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the critic
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the critic network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The critic loss.
        """
        pass

    def save_model(self, path: str) -> None:
        torch.save(self._critic, path + "_critic")
        torch.save(self._actor, path + "_actor")

    def load_model(self, path: str) -> None:
        self._critic = torch.load(
            path + "_critic", map_location="cpu", weights_only=False
        ).to(self.device)
        self._actor = torch.load(
            path + "_actor", map_location="cpu", weights_only=False
        ).to(self.device)

    def compute_f_value(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Computes the f value for a batch of transitions.
        Args:
            batch (TransitionBatch): A batch of transitions.
        Returns:
            f_value (Tensor): The value function for the batch of transitions.
        """
        qs = self._critic_target.get_q_values(
            batch.state, batch.action, get_all_values=True
        )
        return torch.mean(qs).detach()
