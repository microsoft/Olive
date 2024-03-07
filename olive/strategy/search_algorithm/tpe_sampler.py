# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict

import optuna

from olive.common.config_utils import ConfigParam
from olive.strategy.search_algorithm.optuna_sampler import OptunaSearchAlgorithm


class TPESearchAlgorithm(OptunaSearchAlgorithm):
    """Sample using TPE (Tree-structured Parzen Estimator) algorithm. Uses optuna TPESampler underneath.

    Refer to https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
    for more details about the sampler.
    """

    name = "tpe"

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            **super()._default_config(),
            "multivariate": ConfigParam(
                type_=bool, default_value=True, description="Use multivariate TPE when suggesting parameters."
            ),
            "group": ConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "If this and multivariate are True, the multivariate TPE with the group decomposed search space is"
                    " used when suggesting parameters. Refer to 'group' at"
                    " https://optuna.readthedocs.io/en/stable/reference/samplers/generated/"
                    "optuna.samplers.TPESampler.html for more information."
                ),
            ),
        }

    def _create_sampler(self) -> optuna.samplers.TPESampler:
        """Create the sampler."""
        return optuna.samplers.TPESampler(
            multivariate=self._config.multivariate, group=self._config.group, seed=self._config.seed
        )
