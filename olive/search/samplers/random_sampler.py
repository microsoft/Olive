# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from random import Random
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from olive.common.config_utils import ConfigBase, ConfigParam
from olive.search.samplers.search_sampler import SearchSampler
from olive.search.search_space import SearchSpace

if TYPE_CHECKING:
    from olive.search.search_point import SearchPoint


class RandomSampler(SearchSampler):
    """Random sampler. Samples random points from the search space."""

    name = "random"

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            **super()._default_config(),
            "seed": ConfigParam(type_=int, default_value=1, description="Seed for the rng."),
        }

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Union[Dict[str, Any], ConfigBase]] = None,
        objectives: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__(search_space, config, objectives)

        self._rng = Random(self.config.seed)
        self._search_points = [None] * len(self._search_space)
        self._available = list(range(len(self._search_space)))

    def reset_rng(self):
        """Reset the random number generator."""
        self._rng = Random(self.config.seed)

    @property
    def num_samples_suggested(self) -> int:
        """Returns the number of samples suggested so far."""
        return len(self._search_space) - len(self._available)

    @property
    def should_stop(self) -> bool:
        """Check if the searcher should stop at the current trial."""
        return super().should_stop or (len(self._available) == 0)

    def suggest(self) -> "SearchPoint":
        """Suggest a new configuration to try."""
        if self.should_stop:
            return None

        index = self._available[self._rng.randrange(len(self._available))]
        self._available.remove(index)
        self._search_points[index] = self._search_space[index]
        return self._search_points[index]
