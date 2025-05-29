# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import ClassVar

from olive.common.pydantic_v1 import BaseModel
from olive.data.component.dataloader import default_calibration_dataloader
from olive.data.config import DataConfig, DefaultDataComponentCombos
from olive.data.constants import DataContainerType, DefaultDataContainer
from olive.data.registry import Registry


@Registry.register(DataContainerType.DATA_CONTAINER, name=DefaultDataContainer.DATA_CONTAINER.value)
class DataContainer(BaseModel):
    """Base class for data container."""

    # override the default components from config with baseclass or subclass
    default_components_type: ClassVar[dict] = DefaultDataComponentCombos
    # avoid to directly create the instance of DataComponentConfig,
    # suggest to use config.to_data_container()
    config: DataConfig = None

    def load_dataset(self):
        """Run load dataset."""
        return self.config.load_dataset(**self.config.load_dataset_params)

    def pre_process(self, dataset):
        """Run pre_process."""
        return self.config.pre_process(dataset, **self.config.pre_process_params)

    def post_process(self, output_data):
        """Run post_process."""
        return self.config.post_process(output_data, **self.config.post_process_params)

    def dataloader(self, dataset):
        """Run dataloader."""
        return self.config.dataloader(dataset, **self.config.dataloader_params)

    def create_dataloader(self):
        """Create dataloader.

        dataset -> preprocess -> dataloader
        """
        dataset = self.load_dataset()
        pre_process_dataset = self.pre_process(dataset)
        return self.dataloader(pre_process_dataset)

    def create_calibration_dataloader(self, model_path=None, io_config=None, calibration_providers=None):
        """Create calibration dataloader."""
        dataloader = self.create_dataloader()
        return default_calibration_dataloader(
            dataloader, model_path=model_path, io_config=io_config, calibration_providers=calibration_providers
        )

    def get_first_batch(self, dataloader=None):
        """Get first batch of dataloader."""
        dataloader = dataloader or self.create_dataloader()
        return next(iter(dataloader))

    def update_component(self):
        return None
