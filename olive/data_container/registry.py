# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Union

from olive.data_container.constants import DataComponentType, DataContainerType, DefaultDataContainer

logger = logging.getLogger(__name__)


class Registry:
    """
    Registry for data components and data containers
    """

    _REGISTRY = {
        DataComponentType.DATASET: {},
        DataComponentType.PRE_PROCESS: {},
        DataComponentType.POST_PROCESS: {},
        DataComponentType.DATALOADER: {},
        DataContainerType.DATA_CONTAINER: {},
    }

    @classmethod
    def register(cls, sub_type: Union[DataComponentType, DataContainerType], name: str = None):
        """
        Register a component class to the registry

        Args:
            sub_type (DataComponentType): the type of the component
            name (str): the name of the component, is name is None, use the class name

        Returns:
            Callable: the decorator function
        """

        def decorator(component):
            if name is None:
                cls._REGISTRY[sub_type][component.__name__] = component
            else:
                cls._REGISTRY[sub_type][name] = component
            return component

        return decorator

    @classmethod
    def get(cls, sub_type: DataComponentType, name: str):
        """
        Get a component class from the registry

        Args:
            component_type (DataComponentType): the type of the component
            name (str): the name of the component

        Returns:
            Type: the component class
        """
        return cls._REGISTRY[sub_type][name]

    @classmethod
    def get_component(cls, component, name: str):
        """ """
        return cls._REGISTRY[component][name]

    @classmethod
    def get_dataset_component(cls, name: str):
        """
        Get a dataset component class from the registry

        Args:
            name (str): the name of the component

        Returns:
            Type: the dataset component class
        """
        return cls.get_component(DataComponentType.DATASET, name)

    @classmethod
    def get_pre_process_component(cls, name: str):
        """
        Get a pre-process component class from the registry

        Args:
            name (str): the name of the component

        Returns:
            Type: the pre-process component class
        """
        return cls.get_component(DataComponentType.PRE_PROCESS, name)

    @classmethod
    def get_post_process_component(cls, name: str):
        """
        Get a post-process component class from the registry

        Args:
            name (str): the name of the component

        Returns:
            Type: the post-process component class
        """
        return cls.get_component(DataComponentType.POST_PROCESS, name)

    @classmethod
    def get_dataloader_component(cls, name: str):
        """
        Get a dataloader component class from the registry

        Args:
            name (str): the name of the component

        Returns:
            Type: the dataloader component class
        """
        return cls.get_component(DataComponentType.DATALOADER, name)

    @classmethod
    def get_container(cls, name: str):
        """
        Get all data container classes from the registry

        Returns:
            Dict[str, Type]: the data container classes
        """
        name = name or DefaultDataContainer.DATA_CONTAINER.value
        return cls._REGISTRY[DataContainerType.DATA_CONTAINER][name]
