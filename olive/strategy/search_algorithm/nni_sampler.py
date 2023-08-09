# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import Any, Dict

from olive.strategy.search_algorithm.search_algorithm import SearchAlgorithm


class NNISearchAlgorithm(SearchAlgorithm):
    """
    Base class for search algorithms that use NNI tuners.
    """

    name = "nni_search_algorithm"

    def initialize(self):
        self._tuner = self._create_tuner()
        self._tuner.set_time_limit(infinite=True)

    # Method to determine if the search should stop
    def should_stop(self):
        pass

    def suggest(self) -> Dict[str, Dict[str, Any]]:
        if self.should_stop():

            return None
        params = self._tuner.generate_parameters()
        return params

    def report(self, search_point: Dict[str, Dict[str, Any]], result: Dict[str, float], should_prune: bool = False):
        # Logic to report results back to the tuner goes here
        pass

    @abstractmethod
    def _create_tuner(self):
        """
        Create the tuner.
        """
        pass
