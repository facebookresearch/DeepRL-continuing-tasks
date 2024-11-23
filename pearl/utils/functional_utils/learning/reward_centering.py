# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class TD_RC:
    """
    TD-based reward centering.
    """

    def __init__(self, optimizer, initialize_reward_rate) -> None:
        self.optimizer = optimizer
        self.initialize_reward_rate = initialize_reward_rate
        self.init_reward_rate_learning_rate = optimizer.param_groups[0]["lr"]


class RVI_RC:
    """
    RVI-based reward centering.
    """

    def __init__(self, ref_states_update_freq) -> None:
        assert ref_states_update_freq >= 1
        self.ref_states_update_freq = ref_states_update_freq
        self.f_batch = None


class MA_RC:
    """
    Moving average-based reward centering.
    """

    def __init__(self, ma_rate) -> None:
        self.ma_rate = ma_rate
