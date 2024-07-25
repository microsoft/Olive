# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
from typing import ClassVar, Dict, Union

from olive.data.constants import DataComponentType, DataContainerType, DefaultDataComponent, DefaultDataContainer

logger = logging.getLogger(__name__)


class Registry:
    """Registry for data components and data containers.

    All component names are case insensitive and stored in lower case.
    """

    _REGISTRY: ClassVar[Dict] = {
        DataComponentType.LOAD_DATASET.value: {},
        DataComponentType.PRE_PROCESS_DATA.value: {},
        DataComponentType.POST_PROCESS_DATA.value: {},
        DataComponentType.DATALOADER.value: {},
        DataContainerType.DATA_CONTAINER.value: {},
    }

    @classmethod
    def register(cls, sub_type: Union[DataComponentType, DataContainerType], name: str = None):
        """Register a component class to the registry.

        Args:
            sub_type (DataComponentType): the type of the component
            name (str): the name of the component, is name is None, use the class name

        Returns:
            Callable: the decorator function

        """

        def decorator(component):
            # make the component name case insensitive
            component_name = (name if name is not None else component.__name__).lower()
            if component_name in cls._REGISTRY[sub_type.value]:
                component_1 = cls._REGISTRY[sub_type.value][component_name]
                component_2 = component

                component_file_1 = inspect.getfile(component_1)
                component_file_2 = inspect.getfile(component_2)

                _, component_line_no_1 = inspect.getsourcelines(component_1)
                _, component_line_no_2 = inspect.getsourcelines(component_2)

                if (component_file_1 != component_file_2) or (component_line_no_1 != component_line_no_2):
                    logger.critical(
                        "%s: Duplicate component registration.\n"
                        "\tPrevious Registration: %s:%d\n"
                        "\tCurrent Registration: %s:%d.",
                        component_name,
                        component_file_1,
                        component_line_no_1,
                        component_file_2,
                        component_line_no_2,
                    )
            cls._REGISTRY[sub_type.value][component_name] = component
            return component

        return decorator

    @classmethod
    def register_dataset(cls, name: str = None):
        """Register a dataset component class to the registry.

        Args:
            name (str): the name of the component, is name is None, use the class name

        Returns:
            Callable: the decorator function

        """
        return cls.register(DataComponentType.LOAD_DATASET, name)

    @classmethod
    def register_pre_process(cls, name: str = None):
        """Register a pre-process component class to the registry.

        Args:
            name (str): the name of the component, is name is None, use the class name

        Returns:
            Callable: the decorator function

        """
        return cls.register(DataComponentType.PRE_PROCESS_DATA, name)

    @classmethod
    def register_post_process(cls, name: str = None):
        """Register a post-process component class to the registry.

        Args:
            name (str): the name of the component, is name is None, use the class name

        Returns:
            Callable: the decorator function

        """
        return cls.register(DataComponentType.POST_PROCESS_DATA, name)

    @classmethod
    def register_dataloader(cls, name: str = None):
        """Register a dataloader component class to the registry.

        Args:
            name (str): the name of the component, is name is None, use the class name

        Returns:
            Callable: the decorator function

        """
        return cls.register(DataComponentType.DATALOADER, name)

    @classmethod
    def register_default_dataset(cls):
        """Register the default dataset component class to the registry.

        Returns:
            Callable: the decorator function

        """
        return cls.register_dataset(DefaultDataComponent.LOAD_DATASET.value)

    @classmethod
    def register_default_pre_process(cls):
        """Register the default pre-process component class to the registry.

        Returns:
            Callable: the decorator function

        """
        return cls.register_pre_process(DefaultDataComponent.PRE_PROCESS_DATA.value)

    @classmethod
    def register_default_post_process(cls):
        """Register the default post-process component class to the registry.

        Returns:
            Callable: the decorator function

        """
        return cls.register_post_process(DefaultDataComponent.POST_PROCESS_DATA.value)

    @classmethod
    def register_default_dataloader(cls):
        """Register the default dataloader component class to the registry.

        Returns:
            Callable: the decorator function

        """
        return cls.register_dataloader(DefaultDataComponent.DATALOADER.value)

    @classmethod
    def get(cls, sub_type: DataComponentType, name: str):
        """Get a component class from the registry.

        Args:
            sub_type (DataComponentType): the type of the component
            name (str): the name of the component

        Returns:
            Type: the component class

        """
        return cls._REGISTRY[sub_type][name.lower()]

    @classmethod
    def get_component(cls, component: str, name: str):
        return cls._REGISTRY[component][name.lower()]

    @classmethod
    def get_load_dataset_component(cls, name: str):
        """Get a dataset component class from the registry.

        Args:
            name (str): the name of the component

        Returns:
            Type: the dataset component class

        """
        return cls.get_component(DataComponentType.LOAD_DATASET.value, name)

    @classmethod
    def get_pre_process_component(cls, name: str):
        """Get a pre-process component class from the registry.

        Args:
            name (str): the name of the component

        Returns:
            Type: the pre-process component class

        """
        return cls.get_component(DataComponentType.PRE_PROCESS_DATA.value, name)

    @classmethod
    def get_post_process_component(cls, name: str):
        """Get a post-process component class from the registry.

        Args:
            name (str): the name of the component

        Returns:
            Type: the post-process component class

        """
        return cls.get_component(DataComponentType.POST_PROCESS_DATA.value, name)

    @classmethod
    def get_dataloader_component(cls, name: str):
        """Get a dataloader component class from the registry.

        Args:
            name (str): the name of the component

        Returns:
            Type: the dataloader component class

        """
        return cls.get_component(DataComponentType.DATALOADER.value, name)

    @classmethod
    def get_container(cls, name: str):
        """Get all data container classes from the registry.

        Returns:
            Dict[str, Type]: the data container classes

        """
        name = name or DefaultDataContainer.DATA_CONTAINER.value
        return cls._REGISTRY[DataContainerType.DATA_CONTAINER.value][name.lower()]

    @classmethod
    def get_default_load_dataset_component(cls):
        """Get the default dataset component class from the registry.

        Returns:
            Type: the default dataset component class

        """
        return cls.get_load_dataset_component(DefaultDataComponent.LOAD_DATASET.value)

    @classmethod
    def get_default_pre_process_component(cls):
        """Get the default pre-process component class from the registry.

        Returns:
            Type: the default pre-process component class

        """
        return cls.get_pre_process_component(DefaultDataComponent.PRE_PROCESS_DATA.value)

    @classmethod
    def get_default_post_process_component(cls):
        """Get the default post-process component class from the registry.

        Returns:
            Type: the default post-process component class

        """
        return cls.get_post_process_component(DefaultDataComponent.POST_PROCESS_DATA.value)

    @classmethod
    def get_default_dataloader_component(cls):
        """Get the default dataloader component class from the registry.

        Returns:
            Type: the default dataloader component class

        """
        return cls.get_dataloader_component(DefaultDataComponent.DATALOADER.value)
