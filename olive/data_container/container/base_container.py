# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pydantic import BaseModel

from olive.data_container.config import DataContainerConfig, DefaultDataComponentCombos
from olive.data_container.constants import DataContainerType, DefaultDataContainer
from olive.data_container.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER, name=DefaultDataContainer.DATA_CONTAINER.value)
class BaseContainer(BaseModel):
    """
    Base class for data containers.
    """

    # override the default components from config with baseclass or subclass
    default_components_type: dict = DefaultDataComponentCombos
    config: DataContainerConfig = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = DataContainerConfig(default_components_type=self.default_components_type)

    def dataset(self):
        """
        Run dataset
        """
        return self.config.dataset(self.config.dataset_params)

    def pre_process(self):
        """
        Run pre_process
        """
        return self.config.pre_process(self.config.pre_process_params)

    def post_process(self):
        """
        Run post_process
        """
        return self.config.post_process(self.config.post_process_params)

    def dataloader(self):
        """
        Run dataloader
        """
        return self.config.dataloader(self.config.dataloader_params)
