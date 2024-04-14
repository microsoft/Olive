# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict

from olive.common.config_utils import ConfigParam
from olive.strategy.search_algorithm.search_algorithm import SearchAlgorithm


class RandomSearchAlgorithm(SearchAlgorithm):
    """Random Searcher. Samples random points from the search space with or without replacement."""

    name = "random"

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            "num_samples": ConfigParam(type_=int, default_value=1, description="Number of samples to suggest."),
            "seed": ConfigParam(type_=int, default_value=1, description="Seed for the rng."),
            "with_replacement": ConfigParam(type_=bool, default_value=False, description="Sample with replacement."),
        }

    def initialize(self):
        """Initialize the searcher."""
        self._search_space.set_seed(self._config.seed)
        if not self._config.with_replacement:
            self._options = list(self._search_space.iterate())
        self._num_samples_suggested = 0

    def should_stop(self):
        should_stop = (self._search_space.empty() and self._num_samples_suggested > 0) or (
            self._num_samples_suggested >= self._config.num_samples
        )
        if not self._config.with_replacement:
            should_stop = should_stop or (len(self._options) == 0)
        return should_stop or super().should_stop()

    def suggest(self) -> Dict[str, Dict[str, Any]]:
        """Suggest a new configuration to try."""
        if self.should_stop():
            return None

        if self._config.with_replacement:
            # sample a randrom point from the search space with replacement
            search_point = self._search_space.random_sample()
        else:
            # sample a random point from the search space without replacement
            search_point = self._search_space.rng.choice(self._options)
            self._options.remove(search_point)

        self._num_samples_suggested += 1

        return search_point

    def report(self, search_point: Dict[str, Dict[str, Any]], result: Dict[str, Any], should_prune: bool = False):
        """Report the result of a configuration."""
