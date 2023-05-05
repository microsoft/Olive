# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import ClassVar

from pydantic import BaseModel

from olive.data_config.component.dataloader import default_calibration_dataloader
from olive.data_config.config import DataConfig, DefaultDataComponentCombos
from olive.data_config.constants import DataContainerType, DefaultDataContainer
from olive.data_config.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER, name=DefaultDataContainer.DATA_CONTAINER.value)
class BaseContainer(BaseModel):
    """
    Base class for data containers.
    """

    # override the default components from config with baseclass or subclass
    default_components_type: ClassVar[dict] = DefaultDataComponentCombos
    # avoid to directly create the instance of DataComponentConfig,
    # suggest to use config.to_data_container()
    config: DataConfig = None

    # not be used, for read only. when you update the components function,
    # please update the _params_list. It should be key name of params_config
    _params_list: list = [
        "data_dir",
        "label_cols",
        "batch_size",
    ]

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.config = DataConfig(
    #         type=self.__class__.__name__,
    #         default_components_type=self.default_components_type,
    #     )

    def dataset(self):
        """
        Run dataset
        """
        return self.config.dataset(**self.config.dataset_params)

    def pre_process(self, dataset):
        """
        Run pre_process
        """
        return self.config.pre_process(dataset, **self.config.pre_process_params)

    def post_process(self, output_data):
        """
        Run post_process
        """
        return self.config.post_process(output_data, **self.config.post_process_params)

    def dataloader(self, dataset):
        """
        Run dataloader
        """
        return self.config.dataloader(dataset, **self.config.dataloader_params)

    def create_dataloader(self):
        """
        Create dataloader
        dataset -> preprocess -> dataloader
        """
        dataset = self.dataset()
        pre_process_dataset = self.pre_process(dataset)
        return self.dataloader(pre_process_dataset)

    def create_calibration_dataloader(self):
        """
        Create calibration dataloader
        """
        dataloader = self.create_dataloader()
        return default_calibration_dataloader(dataloader)

    def update_component(self):
        return None
