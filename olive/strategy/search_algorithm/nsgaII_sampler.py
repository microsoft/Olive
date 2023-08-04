# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict

import optuna

from olive.common.config_utils import ConfigParam
from olive.strategy.search_algorithm.optuna_sampler import OptunaSearchAlgorithm


class NSGAIISearchAlgorithm(OptunaSearchAlgorithm):
    """
    Sample using NSGAII (Nondominated Sorting Genetic Algorithm II) algorithm. Uses optuna NSGAIISampler underneath.

    Refer to https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html
    for more details about the sampler.
    """

    name = "nsgaII"

    @staticmethod
    def _default_config() -> Dict[str, ConfigParam]:
        return {
            **OptunaSearchAlgorithm._default_config(),
            "population_size": ConfigParam(
                type_=int, default_value=50, description="Number of individuals (trials) in a generation"
            ),
            "mutation_prob": ConfigParam(
                type_=float,
                default_value=None,
                description="Probability of mutating each parameter when creating a new individual.",
            ),
            "crossover_prob": ConfigParam(
                type_=float,
                default_value=0.9,
                description="Probability that a crossover will occur when creating a new individual.",
            ),
            "swapping_prob": ConfigParam(
                type_=float,
                default_value=0.5,
                description="Probability of swapping each parameter of the parents during crossover.",
            ),
            "seed": ConfigParam(
                type_=int,
                default_value=None,
                description="Seed for random number generator. If not specified, a random seed is used.",
            ),
            "crossover": ConfigParam(
                type_=str,
                default_value=None,
                description="Crossover to be applied when creating child individuals.",
            ),
        }

    def _create_sampler(self) -> optuna.samplers.NSGAIISampler:
        """
        Create the sampler.
        """
        return optuna.samplers.NSGAIISampler(
            seed=self._config.seed,
            population_size=self._config.population_size,
            mutation_prob=self._config.mutation_prob,
            crossover_prob=self._config.crossover_prob,
            swapping_prob=self._config.swapping_prob,
            crossover=self._config.crossover,
        )
