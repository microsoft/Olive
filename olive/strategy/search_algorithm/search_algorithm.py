# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from olive.common.auto_config import AutoConfigClass
from olive.common.config_utils import ConfigBase
from olive.strategy.search_parameter import SearchParameter
from olive.strategy.search_space import SearchSpace


class SearchAlgorithm(AutoConfigClass):
    """
    Abstract base class for searchers
    """

    registry: ClassVar[Dict[str, Type["SearchAlgorithm"]]] = {}

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        raise NotImplementedError

    def __init__(
        self,
        search_space: dict[str, dict[str, SearchParameter]],
        objectives: Optional[list[str]] = None,
        higher_is_betters: Optional[list[bool]] = None,
        config: Optional[Union[dict[str, Any], ConfigBase]] = None,
    ):
        # search space
        self._search_space = SearchSpace(search_space)
        if self._search_space.size() == 0:
            raise ValueError("There are no valid points in the search space.")

        # objectives and directions
        objectives = objectives or []
        higher_is_betters = higher_is_betters or []
        assert len(objectives) == len(higher_is_betters), "Number of objectives must match number of higher_is_betters"
        self._objectives = objectives
        self._higher_is_betters = higher_is_betters

        super().__init__(config)
        # TODO: Stop using _ private methods like _objectives, _config, etc
        self._config = self.config
        self.initialize()

    @abstractmethod
    def initialize(self):
        """
        Initialize the searcher.
        """
        pass

    def should_stop(self):
        """
        Check if the searcher should prune the current trial.
        """
        return False

    @abstractmethod
    def suggest(self) -> dict[str, dict[str, Any]]:
        """
        Suggest a new configuration to try.
        """
        pass

    @abstractmethod
    def report(
        self, search_point: dict[str, dict[str, Any]], result: dict[str, Union[float, int]], should_prune: bool = False
    ):
        """
        Report the result of a configuration.
        """
        pass
