# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import torch
import torch.nn as nn

"""
This file is a collection of some functions used to create and update critic networks
as well as compute optimization losses.
"""
# TODO 1: see if we can remove the `update_critic_target_networks` and
# `single_critic_state_value_loss` functions.

# TODO 2: see if we can add functions for updating the target networks and computing losses
# in the `EnsembleQValueNetwork` class.


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
