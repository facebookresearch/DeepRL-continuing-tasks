# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Activations and loss functions
# TODO: Make these into Enums
ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softmax": nn.Softmax,
}

LOSS_TYPES = {
    "mse": nn.functional.mse_loss,
    "mae": nn.functional.l1_loss,
    "cross_entropy": nn.functional.binary_cross_entropy,
}


def mlp_block(
    input_dim: int,
    hidden_dims: Optional[List[int]],
    output_dim: int = 1,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    hidden_activation: str = "relu",
    last_activation: Optional[str] = None,
    dropout_ratio: float = 0.0,
    use_skip_connections: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    A simple MLP which can be reused to create more complex networks
    Args:
        input_dim: dimension of the input layer
        hidden_dims: a list of dimensions of the hidden layers
        output_dim: dimension of the output layer
        use_batch_norm: whether to use batch_norm or not in the hidden layers
        hidden_activation: activation function used for hidden layers
        last_activation: this is optional, if need activation for layer, set this input
                        otherwise, no activation is applied on last layer
        dropout_ratio: user needs to call nn.Module.eval to ensure dropout is ignored during act
    Returns:
        an nn.Sequential module consisting of mlp layers
    """
    if hidden_dims is None:
        hidden_dims = []
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(dims) - 2):
        single_layers = []
        input_dim_current_layer = dims[i]
        output_dim_current_layer = dims[i + 1]
        single_layers.append(
            nn.Linear(input_dim_current_layer, output_dim_current_layer)
        )
        if use_layer_norm:
            single_layers.append(nn.LayerNorm(output_dim_current_layer))
        if dropout_ratio > 0:
            single_layers.append(nn.Dropout(p=dropout_ratio))
        single_layers.append(ACTIVATION_MAP[hidden_activation]())
        if use_batch_norm:
            single_layers.append(nn.BatchNorm1d(output_dim_current_layer))
        single_layer_model = nn.Sequential(*single_layers)
        if use_skip_connections:
            if input_dim_current_layer == output_dim_current_layer:
                single_layer_model = ResidualWrapper(single_layer_model)
            else:
                logging.warning(
                    "Skip connections are enabled, "
                    f"but layer in_dim ({input_dim_current_layer}) != out_dim "
                    f"({output_dim_current_layer})."
                    "Skip connection will not be added for this layer"
                )
        layers.append(single_layer_model)

    last_layer = []
    last_layer.append(nn.Linear(dims[-2], dims[-1]))
    if last_activation is not None:
        last_layer.append(ACTIVATION_MAP[last_activation]())
    last_layer_model = nn.Sequential(*last_layer)
    if use_skip_connections:
        if dims[-2] == dims[-1]:
            last_layer_model = ResidualWrapper(last_layer_model)
        else:
            logging.warning(
                "Skip connections are enabled, "
                f"but layer in_dim ({dims[-2]}) != out_dim ({dims[-1]}). "
                "Skip connection will not be added for this layer"
            )
    layers.append(last_layer_model)
    return nn.Sequential(*layers)


def conv_block(
    input_channels_count: int,
    output_channels_list: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
    use_batch_norm: bool = False,
) -> nn.Module:
    """
    Reminder: torch.Conv2d layers expect inputs as (batch_size, in_channels, height, width)
    Notes: layer norm is typically not used with CNNs

    Args:
        input_channels_count: number of input channels
        output_channels_list: a list of number of output channels for each convolutional layer
        kernel_sizes: a list of kernel sizes for each layer
        strides: a list of strides for each layer
        paddings: a list of paddings for each layer
        use_batch_norm: whether to use batch_norm or not in the convolutional layers
    Returns:
        an nn.Sequential module consisting of convolutional layers
    """
    layers = []
    for out_channels, kernel_size, stride, padding in zip(
        output_channels_list, kernel_sizes, strides, paddings
    ):
        conv_layer = nn.Conv2d(
            input_channels_count,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        layers.append(conv_layer)
        if use_batch_norm and input_channels_count > 1:
            layers.append(
                nn.BatchNorm2d(input_channels_count)
            )  # input to Batchnorm 2d is the number of input channels
        layers.append(nn.ReLU())
        # number of input channels to next layer is number of output channels of previous layer:
        input_channels_count = out_channels

    return nn.Sequential(*layers)


def xavier_init_weights(m: nn.Module) -> None:
    if hasattr(m, "weight"):
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, "bias"):
        m.bias.data.fill_(0.01)


def kaiming_normal_init_weights(m: nn.Module) -> None:
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight)
    if hasattr(m, "bias"):
        m.bias.data.fill_(0.0)


def orthogonal_init_weights(m: nn.Module) -> None:
    if hasattr(m, "weight"):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
    if hasattr(m, "bias"):
        nn.init.constant_(m.bias, 0)


def update_target_network(
    target_network: nn.Module, source_network: nn.Module, tau: float
) -> None:
    # Q_target = (1 - tao) * Q_target + tao*Q
    for target_param, source_param in zip(
        target_network.parameters(), source_network.parameters()
    ):
        if target_param is source_param:
            # skip soft-updating when the target network shares the parameter with
            # the network being train.
            continue
        new_param = tau * source_param.data + (1.0 - tau) * target_param.data
        target_param.data.copy_(new_param)


def compute_output_dim_model_cnn(
    input_channels: int, input_width: int, input_height: int, model_cnn: nn.Module
) -> int:
    dummy_input = torch.zeros(1, input_channels, input_width, input_height)
    dummy_output_flattened = torch.flatten(
        model_cnn(dummy_input), start_dim=1, end_dim=-1
    )
    return dummy_output_flattened.shape[1]
