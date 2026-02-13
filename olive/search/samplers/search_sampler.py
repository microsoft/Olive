# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

from olive.common.config_utils import ConfigBase, ConfigParam, create_config_class
from olive.search.search_space import SearchSpace

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.search.search_point import SearchPoint


class SearchSampler:
    """Base class for search samplers."""

    registry: ClassVar[dict[str, type["SearchSampler"]]] = {}
    name: Optional[str] = None
    _is_base_class: bool = True

    def __init__(
        self,
        search_space: SearchSpace,
        config: Optional[Union[dict[str, Any], ConfigBase]] = None,
        objectives: dict[str, dict[str, Any]] = None,
    ):
        config = config or {}
        if isinstance(config, dict):
            config = self.get_config_class()(**config)
        self.config = config

        self._search_space = search_space
        self._config = config

        # Order the objectives based on priority, and then by name
        objectives = objectives or {}
        self._objectives = OrderedDict(sorted(objectives.items(), key=lambda entry: (entry[1]["priority"], entry[0])))
        self._higher_is_betters = {
            name: objective.get("higher_is_better") or False for name, objective in self._objectives.items()
        }

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the search sampler."""
        super().__init_subclass__(**kwargs)
        # Subclasses should not be considered base classes unless explicitly set
        if '_is_base_class' not in cls.__dict__:
            cls._is_base_class = False
        if cls._is_base_class:
            return
        name = cls.name if cls.name is not None else cls.__name__.lower()
        cls.registry[name] = cls

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        """Get the default configuration for the sampler.
        
        Subclasses can override this to add more configuration parameters.
        """
        return {
            "max_samples": ConfigParam(
                type_=int,
                default_value=0,
                description="Maximum number of samples to suggest. Search exhaustively if set to zero.",
            ),
        }

    @classmethod
    def get_config_class(cls) -> type[ConfigBase]:
        """Get the configuration class."""
        if '_is_base_class' not in cls.__dict__:
            cls._is_base_class = False
        if cls._is_base_class:
            raise TypeError(f"Cannot get config class for base class {cls.__name__}")
        return create_config_class(f"{cls.__name__}Config", cls._default_config(), ConfigBase, {})

    @property
    def num_samples_suggested(self) -> int:
        """Returns the number of samples suggested so far.
        
        Subclasses must implement this property.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement num_samples_suggested")

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

    def suggest(self) -> "SearchPoint":
        """Suggest a new configuration to try.
        
        Subclasses must implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement suggest")

    def record_feedback_signal(self, search_point_index: int, signal: "MetricResult", should_prune: bool = False):
        """Report the result of a configuration."""
