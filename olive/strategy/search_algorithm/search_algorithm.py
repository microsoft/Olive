# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union

from olive.common.auto_config import AutoConfigClass
from olive.common.config_utils import ConfigBase
from olive.strategy.search_space import SearchSpace

if TYPE_CHECKING:
    from olive.strategy.search_parameter import SearchParameter


class SearchAlgorithm(AutoConfigClass):
    """Abstract base class for searchers."""

    name: str = None
    registry: ClassVar[Dict[str, Type["SearchAlgorithm"]]] = {}

    def __init__(
        self,
        search_space: Dict[str, Dict[str, "SearchParameter"]],
        objectives: Optional[List[str]] = None,
        higher_is_betters: Optional[List[bool]] = None,
        config: Optional[Union[Dict[str, Any], ConfigBase]] = None,
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
        # TODO(jambayk): Stop using _ private methods like _objectives, _config, etc
        self._config = self.config
        self.initialize()

    @abstractmethod
    def initialize(self):
        """Initialize the searcher."""

    def should_stop(self):
        """Check if the searcher should prune the current trial."""
        return False

    @abstractmethod
    def suggest(self) -> Dict[str, Dict[str, Any]]:
        """Suggest a new configuration to try."""

    @abstractmethod
    def report(
        self, search_point: Dict[str, Dict[str, Any]], result: Dict[str, Union[float, int]], should_prune: bool = False
    ):
        """Report the result of a configuration."""
