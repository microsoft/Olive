# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from olive.common.config_utils import ConfigBase
from olive.search.samplers.search_sampler import SearchSampler
from olive.search.search_space import SearchSpace

if TYPE_CHECKING:
    from olive.search.search_point import SearchPoint


class SequentialSampler(SearchSampler):
    """Sequential sampler provides search sequential search points."""

    name = "sequential"

    @classmethod
    def _default_config(cls):
        return super()._default_config()

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Union[Dict[str, Any], ConfigBase]] = None,
        objectives: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__(search_space, config, objectives)

        self._index = 0

    @property
    def num_samples_suggested(self) -> int:
        """Returns the number of samples suggested so far."""
        return self._index

    def suggest(self) -> "SearchPoint":
        """Suggest a new configuration to try."""
        if self.should_stop:
            return None

        index = self._index
        self._index += 1

        return self._search_space[index]
