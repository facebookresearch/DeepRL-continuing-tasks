# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from .device import (
    get_default_device,
    get_pearl_device,
)
from .tensor_like import assert_is_tensor_like

__all__ = [
    "assert_is_tensor_like",
    "get_default_device",
    "get_pearl_device",
]
