# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import dataclasses
from dataclasses import dataclass
from typing import Optional, TypeVar

import torch


T = TypeVar("T", bound="Transition")


@dataclass(frozen=False)
class Transition:
    """
    Transition is designed for one single set of data
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    truncated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    next_state: Optional[torch.Tensor] = None

    def to(self: T, device: torch.device) -> T:
        # iterate over all fields, move to correct device
        for f in dataclasses.fields(self.__class__):
            if getattr(self, f.name) is not None:
                super().__setattr__(
                    f.name,
                    torch.as_tensor(getattr(self, f.name)).to(device),
                )
        return self

    @property
    def device(self) -> torch.device:
        return self.state.device


TB = TypeVar("TB", bound="TransitionBatch")


@dataclass(frozen=False)
class TransitionBatch:
    """
    TransitionBatch is designed for data batch
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    truncated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    next_state: Optional[torch.Tensor] = None

    def to(self: TB, device: torch.device) -> TB:
        # iterate over all fields
        for f in dataclasses.fields(self.__class__):
            if getattr(self, f.name) is not None:
                item = getattr(self, f.name)
                item = torch.as_tensor(item, device=device)
                super().__setattr__(
                    f.name,
                    item,
                )
        return self

    @property
    def device(self) -> torch.device:
        """
        The device where the batch lives.
        """
        return self.state.device

    def __len__(self) -> int:
        return self.reward.shape[0]
