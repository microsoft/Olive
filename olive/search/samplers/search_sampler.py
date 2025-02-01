# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Type, Union

from olive.common.auto_config import AutoConfigClass
from olive.common.config_utils import ConfigBase, ConfigParam
from olive.search.search_space import SearchSpace

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.search.search_point import SearchPoint


class SearchSampler(AutoConfigClass):
    """Abstract base class for searchers."""

    registry: ClassVar[Dict[str, Type["SearchSampler"]]] = {}

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            "max_samples": ConfigParam(
                type_=int,
                default_value=0,
                description="Maximum number of samples to suggest. Search exhaustively if set to zero.",
            ),
        }

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Union[Dict[str, Any], ConfigBase]] = None,
        objectives: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self._search_space = search_space
        self._config = config

        # Order the objectives based on priority, and then by name
        objectives = objectives or {}
        self._objectives = OrderedDict(sorted(objectives.items(), key=lambda entry: (entry[1]["priority"], entry[0])))
        self._higher_is_betters = {
            name: objective.get("higher_is_better") or False for name, objective in self._objectives.items()
        }

    @property
    @abstractmethod
    def num_samples_suggested(self) -> int:
        """Returns the number of samples suggested so far."""
        return 0

    @property
    def max_samples(self) -> int:
        """Returns the maximum number of samples to suggest."""
        return self.config.max_samples

    @property
    def should_stop(self) -> bool:
        """Check if the searcher should stop at the current trial."""
        return (
            (len(self._search_space) == 0)
            or (self.num_samples_suggested >= len(self._search_space))
            or ((self.max_samples > 0) and (self.num_samples_suggested >= self.max_samples))
        )

    @abstractmethod
    def suggest(self) -> "SearchPoint":
        """Suggest a new configuration to try."""
        return None

    def record_feedback_signal(self, search_point_index: int, signal: "MetricResult", should_prune: bool = False):
        """Report the result of a configuration."""
