# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import cast, List, Optional, Type, Union

import torch
import torch.nn as nn

from pearl.neural_networks.common.utils import xavier_init_weights
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
    VanillaValueNetwork,
)

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    EnsembleQValueNetwork,
    QValueNetwork,
    VanillaQValueNetwork,
)

"""
This file is a collection of some functions used to create and update critic networks
as well as compute optimization losses.
"""
# TODO 1: see if we can remove the `update_critic_target_networks` and
# `single_critic_state_value_loss` functions.

# TODO 2: see if we can add functions for updating the target networks and computing losses
# in the `EnsembleQValueNetwork` class.


def make_critic(
    state_dim: int,
    hidden_dims: Optional[List[int]],
    ensemble_critic_size: int,  # used only for ensemble critic
    network_type: Union[Type[ValueNetwork], Type[QValueNetwork]],
    action_dim: Optional[int] = None,
) -> nn.Module:
    """
    A utility function to instantiate a critic network. 

    Args:
        state_dim (int): Dimension of the observation space.
        hidden_dims (Optional[Iterable[int]]): Hidden dimensions of the critic network.
        ensemble_critic_size: Number of critics.
            Used only when network_type is EnsembleQValueNetwork.
        network_type (Union[Type[ValueNetwork], Type[QValueNetwork]]): The type of the critic
            network to instantiate.
        action_dim (Optional[int]): The dimension of the action space.

    Returns:
        critic network (nn.Module): The critic network to be used by different modules.
    """
    if network_type == EnsembleQValueNetwork:
        assert action_dim is not None
        assert hidden_dims is not None
        # cast network_type to get around static Pyre type checking; the runtime check with
        # `issubclass` ensures the network type is a sublcass of QValueNetwork
        network_type = cast(Type[QValueNetwork], network_type)

        return EnsembleQValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            ensemble_size=ensemble_critic_size,
            prior_scale=1.0,
            init_fn=xavier_init_weights,
        )
    else:
        if network_type == VanillaQValueNetwork:
            # pyre-ignore[45]:
            # Pyre does not know that `network_type` is asserted to be concrete
            return network_type(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        elif network_type == VanillaValueNetwork:
            # pyre-ignore[45]:
            # Pyre does not know that `network_type` is asserted to be concrete
            return network_type(
                input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1
            )
        else:
            raise NotImplementedError(
                f"Type {network_type} cannot be used to instantiate a critic network."
            )

def ensemble_critic_action_value_loss(
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    critic: nn.Module,
    reward_rate: torch.Tensor,
) -> torch.Tensor:
    """
    This method calculates the sum of the mean squared errors between the predicted Q-values
    using critic networks (LHS of the Bellman equation) and the input target estimates (RHS of the
    Bellman equation).

    Args:
        state_batch (torch.Tensor): A batch of states with expected shape
            `(batch_size, state_dim)`.
        action_batch (torch.Tensor): A batch of actions with expected shape
            `(batch_size, action_dim)`.
        expected_target_batch (torch.Tensor): The batch of target estimates
            (i.e. RHS of the Bellman equation) with expected shape `(batch_size)`.
        critic (Ensemble Critic): The ensemble critic network to update.
    Returns:
        loss (torch.Tensor): Sum of mean squared errors in the Bellman equation (for action-value
            prediction) corresponding to both critic networks. The expected shape is `()`. This
            scalar loss is used to train ensemble critic network.
    """
    criterion = torch.nn.MSELoss()
    qs = critic.get_q_values(
        state_batch,
        action_batch,
        get_all_values=True,
    )  # shape (num_critic, batch_size)
    loss_list = []
    for i in range(qs.shape[0]):
        loss_list.append(
            criterion(
                qs[i].reshape_as(expected_target_batch) + reward_rate,
                expected_target_batch.detach(),
            )
        )
    loss = sum(loss_list)
    assert isinstance(loss, torch.Tensor)
    return loss
