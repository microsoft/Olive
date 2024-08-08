# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
from typing import ClassVar, Dict

logger = logging.getLogger(__name__)


class Registry:
    """Registry for olive model evaluators."""

    _REGISTRY: ClassVar[Dict] = {}

    @classmethod
    def register(cls, name: str = None):
        """Register an evaluator to the registry.

        Args:
            name (str): the name of the evaluator, if name is None, uses the class name

        Returns:
            Callable: the decorator function

        """

        def decorator(component):
            component_name = name if name is not None else component.__name__
            if component_name in cls._REGISTRY:
                component_1 = cls._REGISTRY[component_name]
                component_2 = component

                component_file_1 = inspect.getfile(component_1)
                component_file_2 = inspect.getfile(component_2)

                _, component_line_no_1 = inspect.getsourcelines(component_1)
                _, component_line_no_2 = inspect.getsourcelines(component_2)

                if (component_file_1 != component_file_2) or (component_line_no_1 != component_line_no_2):
                    logger.critical(
                        "%s: Duplicate evaluator registration.\n"
                        "\tPrevious Registration: %s:%d\n"
                        "\tCurrent Registration: %s:%d.",
                        component_name,
                        component_file_1,
                        component_line_no_1,
                        component_file_2,
                        component_line_no_2,
                    )
            cls._REGISTRY[component_name] = component
            return component

        return decorator

    @classmethod
    def get(cls, name: str):
        """Get an evaluator, by name, from the registry.

        Args:
            name (str): the name of the evaluator

        Returns:
            Type: the OliveEvaluator class

        """
        return cls._REGISTRY.get(name)
