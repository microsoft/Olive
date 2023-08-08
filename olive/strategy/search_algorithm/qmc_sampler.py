# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict

import optuna

from olive.common.config_utils import ConfigParam
from olive.strategy.search_algorithm.optuna_sampler import OptunaSearchAlgorithm


class QMCSearchAlgorithm(OptunaSearchAlgorithm):
    """
    Sample using QMC (A Quasi Monte Carlo Sampler) algorithm. Uses optuna QMCSampler underneath.

    Refer to https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html
    for more details about the sampler.
    """

    name = "qmc"

    @staticmethod
    def _default_config() -> Dict[str, ConfigParam]:
        return {
            **OptunaSearchAlgorithm._default_config(),
            "scramble": ConfigParam(
                type_=bool,
                default_value=True,
                description="If this option is True, scrambling (randomization) is applied to the QMC sequences.",
            ),
            "qmc_type": ConfigParam(
                type_=str,
                default_value="sobol",
                description="The type of sequence to be sampled. Available options are 'halton' and 'sobol'.",
            ),
        }

    def _create_sampler(self) -> optuna.samplers.QMCSampler:
        """
        Create the sampler.
        """
        return optuna.samplers.QMCSampler(
            seed=self._config.seed,
            scramble=self._config.scramble,
            qmc_type=self._config.qmc_type,
        )
