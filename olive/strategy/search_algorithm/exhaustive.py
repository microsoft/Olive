# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any

from olive.strategy.search_algorithm.search_algorithm import SearchAlgorithm


class ExhaustiveSearchAlgorithm(SearchAlgorithm):
    """
    Exhaustive Search Algorithm. Does a grid search over the search space.
    """

    name = "exhaustive"

    @staticmethod
    def _default_config():
        return {}

    def initialize(self):
        """
        Initialize the searcher.
        """
        self._iterator = self._search_space.iterate()

    def suggest(self) -> dict[str, dict[str, Any]]:
        """
        Suggest a new configuration to try.
        """
        try:
            return next(self._iterator)
        except StopIteration:
            return None

    def report(self, search_point: dict[str, dict[str, Any]], result: dict[str, Any], should_prune: bool = False):
        """
        Report the result of a configuration.
        """
        return
