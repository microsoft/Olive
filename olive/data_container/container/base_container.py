# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pydantic import BaseModel

from olive.data_container.config import DataContainerConfig
from olive.data_container.constants import DataContainerType, DefaultDataComponent, DefaultDataContainer
from olive.data_container.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER, name=DefaultDataContainer.DATA_CONTAINER.value)
class BaseContainer(BaseModel):
    """
    Base class for data containers.
    """

    config: DataContainerConfig = DataContainerConfig()
    _dataset: str = DefaultDataComponent.DATASET.value
    _pre_process: str = DefaultDataComponent.PRE_PROCESS.value
    _post_process: str = DefaultDataComponent.POST_PROCESS.value
    _dataloder: str = DefaultDataComponent.DATALOADER.value

    def get_params(self, components_name):
        """
        Get the parameters of the data container.

        Args:
            components_name (str): the name of the data container component.
            **kwargs: the keyword arguments.

        Returns:
            dict: the parameters of the data container.
        """
        return self.config.components[components_name].params

    @property
    def dataset(self):
        """
        Get the dataset of the data container.

        Returns:
            Dataset: the dataset of the data container.
        """
        if self.config.components.dataset.name is not None:
            return self.config.components.dataset.name
        return Registry.get_dataset_component(self._dataset)

    @property
    def pre_process(self):
        """
        Get the pre-process of the data container.

        Returns:
            PreProcess: the pre-process of the data container.
        """
        if self.config.components.pre_process.name is not None:
            return self.config.components.pre_process.name
        return Registry.get_pre_process_component(self._pre_process)

    @property
    def post_process(self):
        """
        Get the post-process of the data container.

        Returns:
            PostProcess: the post-process of the data container.
        """
        if self.config.components.post_process.name is not None:
            return self.config.components.post_process.name
        return Registry.get_post_process_component(self._post_process)

    @property
    def dataloder(self):
        """
        Get the dataloader of the data container.

        Returns:
            DataLoader: the dataloader of the data container.
        """
        if self.config.components.dataloader.name is not None:
            return self.config.components.dataloader.name
        return Registry.get_dataloader_component(self._dataloder)
