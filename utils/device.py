# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import torch
import torch.distributed as dist

def get_pearl_device(device_id: int = -1) -> torch.device:
    if device_id != -1:
        return torch.device("cuda:" + str(device_id))

    try:
        # This is to pytorch distributed run, and should not affect
        # original implementation of this file
        local_rank = dist.get_rank()
    except Exception:
        local_rank = 0

    return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


def get_default_device() -> torch.device:
    """
    Returns the torch default device, that is,
    the device on which factory methods without a `device`
    specification place their tensors.
    """
    return torch.tensor(0).device
