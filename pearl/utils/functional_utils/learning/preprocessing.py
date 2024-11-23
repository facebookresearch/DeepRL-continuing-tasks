# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from pearl.api.action_result import ActionResult


# taken from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Tracks the mean, variance and count of values."""
        self.mean: np.ndarray = np.zeros(shape, "float32")
        self.var: np.ndarray = np.ones(shape, "float32")
        self.count: int = 0

    def update(self, x: np.ndarray) -> None:
        """Updates the mean, var and count from a batch of samples."""
        if x.dtype == np.float64:
            x = x.astype(np.float32)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


# taken from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py
def update_mean_var_count_from_moments(
    mean: np.ndarray,
    var: np.ndarray,
    count: int,
    batch_mean: np.ndarray,
    batch_var: np.ndarray,
    batch_count: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean.astype(np.float32), new_var.astype(np.float32), new_count


class Preprocessor(ABC):
    @abstractmethod
    def process(self, action_result: ActionResult) -> None:
        pass


class ObservationNormalization(Preprocessor):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        self._obs_rms = RunningMeanStd(shape=shape)
        self._shape = shape
        self._test_time = False

    def process(self, action_result: ActionResult) -> None:
        # preprocess observations by applying normalization
        # see https://arxiv.org/pdf/2006.05990.pdf
        obs = action_result.observation
        assert type(obs) is np.ndarray
        assert len(obs.shape) == 1
        if self._test_time is False:
            self._obs_rms.update(obs.reshape(1, -1))
        normalized_obs = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8)
        action_result.observation = normalized_obs


class RewardClipping(Preprocessor):

    def process(self, action_result: ActionResult) -> None:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        # clip rewards to {+1, 0, -1}
        action_result.reward = np.sign(action_result.reward).item()
