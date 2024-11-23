# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

"""
This file defines PEARL neural network interafaces
User is free to define their own Q(s, a), but would need to inherit from this interface
"""

from __future__ import annotations

import abc
from typing import List, Optional

import torch
from pearl.neural_networks.common.utils import (
    compute_output_dim_model_cnn,
    conv_block,
    mlp_block,
)
from torch import nn, Tensor


class QValueNetwork(abc.ABC, nn.Module):
    """
    An interface for state-action value (Q-value) estimators (typically, neural networks).
    These are value neural networks with a special method
    for computing the Q-value for a state-action pair.
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Returns state dimention"""
        ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Returns action dimention"""
        ...

    @abc.abstractmethod
    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Returns Q(s, a), given s and a
        Args:
            state_batch (torch.Tensor): a batch of state tensors (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of action tensors (batch_size, action_dim)
        Returns:
            Q-values of (state, action) pairs: (batch_size)
        """
        ...


class VanillaQValueNetwork(QValueNetwork):
    """
    A vanilla version of state-action value (Q-value) network.
    It leverages the vanilla implementation of value networks by
    using the state-action pair as the input for the value network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        use_layer_norm: bool = False,
    ) -> None:
        super(VanillaQValueNetwork, self).__init__()
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim
        self._model: nn.Module = mlp_block(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_q_values(
        self,
        state_batch: Tensor,  # (batch_size x state_dim)
        action_batch: Tensor,  # (batch_size x number of query actions x action_dim) or (batch_size x action_dim)
    ) -> Tensor:
        assert len(action_batch.shape) == 2 or len(action_batch.shape) == 3
        if len(action_batch.shape) == 2:
            x = torch.cat(
                [state_batch, action_batch], dim=-1
            )  # (batch_size x (state_dim + action_dim))
            return self.forward(x).view(-1)  # (batch_size)
        state_batch = torch.repeat_interleave(
            state_batch.unsqueeze(1), action_batch.shape[1], dim=1
        )  # (batch_size x number_of_actions_to_query x state_dim)
        x = torch.cat(
            [state_batch, action_batch], dim=-1
        )  # (batch_size x number_of_actions_to_query x (state_dim + action_dim))
        x = x.view(-1, x.shape[-1])
        output = self.forward(x)  # ([batch_size x number_of_actions_to_query] x 1)
        return output.view(
            state_batch.shape[0], action_batch.shape[1]
        )  # (batch_size x number_of_actions_to_query)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class VanillaQValueMultiHeadNetwork(QValueNetwork):
    """
    A vanilla version of state-action value (Q-value) multi-head network.
    It leverages the vanilla implementation of value networks by
    using the state-action pair as the input for the value network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,  # is number of actions in this class
        hidden_dims: List[int],
        use_layer_norm: bool = False,
    ) -> None:
        super(VanillaQValueMultiHeadNetwork, self).__init__()
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim
        self._model: nn.Module = mlp_block(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_q_values(
        self,
        state_batch: Tensor,  # (batch_size x state_dim)
        action_batch: Tensor,  # (batch_size x number of query actions x action_dim) or # (batch_size x action_dim)
    ) -> Tensor:
        q_values_batch = self.forward(state_batch)  # (batch_size x num actions)
        if len(action_batch.shape) == 2:
            return (q_values_batch * action_batch).sum(-1)  # (batch_size)
        q_values_batch = torch.bmm(
            action_batch.type(
                torch.float32
            ),  # shape: (batch_size, number_of_actions_to_query, action_dim)
            q_values_batch.unsqueeze(-1),  # (batch_size x action_dim x 1)
        )  # (batch_size x number_of_actions_to_query x 1)
        q_values_batch = q_values_batch.squeeze(
            -1
        )  # (batch_size x number_of_actions_to_query)
        return q_values_batch

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class EnsembleQValueNetwork(QValueNetwork):
    r"""A Q-value network that uses the `Ensemble` model."""

    def __init__(
        self,
        models: List[nn.Module],
        ensemble_size: int,
    ) -> None:
        super(EnsembleQValueNetwork, self).__init__()
        self.models = models
        self.ensemble_size = ensemble_size

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        z: Optional[int] = None,
        get_all_values: bool = False,
    ) -> Tensor:
        if get_all_values:
            qs = []
            for i in range(self.ensemble_size):
                qs.append(
                    self.models[i].get_q_values(
                        state_batch=state_batch, action_batch=action_batch
                    )  # (batch_size x number_of_actions_to_query) or (batch_size)
                )
            return torch.stack(
                qs
            )  # (ensemble size x batch_size x number_of_actions_to_query) or (ensemble size x batch_size)
        else:
            assert z is not None
            return self.models[z].get_q_values(
                state_batch=state_batch, action_batch=action_batch
            )  # (batch_size x number_of_actions_to_query) or (batch_size)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class CNNQValueNetwork(QValueNetwork):
    """
    A CNN version of state-action value (Q-value) network.
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: List[int],
        output_channels_list: List[int],
        strides: List[int],
        paddings: List[int],
        action_dim: int,
        hidden_dims_fully_connected: Optional[List[int]] = None,
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
    ) -> None:
        super(CNNQValueNetwork, self).__init__()

        self._input_channels = input_channels_count
        self._input_height = input_height
        self._input_width = input_width
        self._output_channels = output_channels_list
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        if hidden_dims_fully_connected is None:
            self._hidden_dims_fully_connected: List[int] = []
        else:
            self._hidden_dims_fully_connected: List[int] = hidden_dims_fully_connected

        self._use_batch_norm_conv = use_batch_norm_conv
        self._use_batch_norm_fully_connected = use_batch_norm_fully_connected

        self._model_cnn: nn.Module = conv_block(
            input_channels_count=self._input_channels,
            output_channels_list=self._output_channels,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            paddings=self._paddings,
            use_batch_norm=self._use_batch_norm_conv,
        )
        # we concatenate actions to state representations in the mlp block of the Q-value network
        self._mlp_input_dims: int = (
            compute_output_dim_model_cnn(
                input_channels=input_channels_count,
                input_width=input_width,
                input_height=input_height,
                model_cnn=self._model_cnn,
            )
            + action_dim
        )
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=1,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )
        self._state_dim: int = input_channels_count * input_height * input_width
        self._action_dim = action_dim

    def get_q_values(
        self,
        state_batch: Tensor,  # shape: (batch_size, input_channels, input_height, input_width)
        action_batch: Tensor,  # shape: (batch_size, number_of_actions_to_query, action_dim) or (batch_size, action_dim)
    ) -> Tensor:
        batch_size = state_batch.shape[0]
        num_query_actions = action_batch.shape[1]
        state_representation_batch = self._model_cnn(
            state_batch / 255.0
        )  # (batch_size x output_channels[-1] x output_height x output_width)
        state_representation_batch = state_representation_batch.view(
            batch_size, -1
        )  # (batch_size x state dim)
        if len(action_batch.shape) == 2:
            x = torch.cat(
                [state_representation_batch, action_batch], dim=-1
            )  # (batch_size x (state_dim + action_dim))
            return self._model_fc(x).view(-1)  # (batch_size)
        # concatenate actions to state representations and do a forward pass through the mlp_block
        state_representation_batch = torch.repeat_interleave(
            state_representation_batch.unsqueeze(1), num_query_actions, dim=1
        )  # (batch_size x number_of_actions_to_query x state_dim)
        x = torch.cat(
            [state_representation_batch, action_batch], dim=-1
        )  # (batch_size x number_of_actions_to_query x (state_dim + action_dim))
        x = x.view(-1, x.shape[-1])
        q_values_batch = self._model_fc(x).reshape(
            batch_size, num_query_actions
        )  # (batch_size x number_of_actions_to_query)
        return q_values_batch

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class CNNQValueMultiHeadNetwork(QValueNetwork):
    """
    A CNN version of state-action value (Q-value) network.
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: List[int],
        output_channels_list: List[int],
        strides: List[int],
        paddings: List[int],
        action_dim: int,  # number of actions in this class
        hidden_dims_fully_connected: Optional[List[int]] = None,
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
    ) -> None:
        super(CNNQValueMultiHeadNetwork, self).__init__()

        self._input_channels = input_channels_count
        self._input_height = input_height
        self._input_width = input_width
        self._output_channels = output_channels_list
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        if hidden_dims_fully_connected is None:
            self._hidden_dims_fully_connected: List[int] = []
        else:
            self._hidden_dims_fully_connected: List[int] = hidden_dims_fully_connected

        self._use_batch_norm_conv = use_batch_norm_conv
        self._use_batch_norm_fully_connected = use_batch_norm_fully_connected

        self._model_cnn: nn.Module = conv_block(
            input_channels_count=self._input_channels,
            output_channels_list=self._output_channels,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            paddings=self._paddings,
            use_batch_norm=self._use_batch_norm_conv,
        )
        # we concatenate actions to state representations in the mlp block of the Q-value network
        self._mlp_input_dims: int = compute_output_dim_model_cnn(
            input_channels=input_channels_count,
            input_width=input_width,
            input_height=input_height,
            model_cnn=self._model_cnn,
        )
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=action_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )
        self._state_dim: int = input_channels_count * input_height * input_width
        self._action_dim = action_dim

    def get_q_values(
        self,
        state_batch: Tensor,  # shape: (batch_size, input_channels, input_height, input_width)
        action_batch: Tensor,  # shape: (batch_size, number_of_actions_to_query, action_dim) or (batch_size, action_dim)
    ) -> Tensor:
        batch_size = state_batch.shape[0]
        state_representation_batch = self._model_cnn(
            state_batch / 255.0
        )  # (batch_size x output_channels[-1] x output_height x output_width)
        state_representation_batch = state_representation_batch.view(
            batch_size, -1
        )  # (batch_size x state dim)
        q_values_batch = self._model_fc(
            state_representation_batch
        )  # (batch_size x num actions)
        if len(action_batch.shape) == 2:
            return (q_values_batch * action_batch).sum(dim=1)  # (batch_size)
        # action representation is assumed to be one hot, so that action_dim = number of actions
        q_values_batch = torch.bmm(
            action_batch.type(
                torch.float32
            ),  # shape: (batch_size, number_of_actions_to_query, action_dim)
            q_values_batch.unsqueeze(-1),  # (batch_size x num actions x 1)
        )  # (batch_size x number_of_actions_to_query x 1)
        q_values_batch = q_values_batch.squeeze(
            -1
        )  # (batch_size x number_of_actions_to_query)
        return q_values_batch

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim
