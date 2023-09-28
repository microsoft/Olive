# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import ClassVar, Optional

from pydantic import BaseModel

from olive.cache import get_local_path_from_root
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

    # not be used, for read only. when you update the components function,
    # please update the _params_list. It should be key name of params_config
    _params_list: list = [
        "data_dir",
        "label_cols",
        "batch_size",
    ]

    def load_dataset(self, data_root_path: Optional[str] = None):
        """Run load dataset."""
        params_config = self.config.load_dataset_params
        data_dir = params_config.get("data_dir")
        data_dir = get_local_path_from_root(data_root_path, data_dir)
        if data_dir:
            params_config["data_dir"] = data_dir
        return self.config.load_dataset(**params_config)

    def pre_process(self, dataset):
        """Run pre_process."""
        return self.config.pre_process(dataset, **self.config.pre_process_params)

    def post_process(self, output_data):
        """Run post_process."""
        return self.config.post_process(output_data, **self.config.post_process_params)

    def dataloader(self, dataset):
        """Run dataloader."""
        return self.config.dataloader(dataset, **self.config.dataloader_params)

    def create_dataloader(self, data_root_path=None):
        """Create dataloader.

        dataset -> preprocess -> dataloader
        """
        dataset = self.load_dataset(data_root_path=data_root_path)
        pre_process_dataset = self.pre_process(dataset)
        return self.dataloader(pre_process_dataset)

    def create_calibration_dataloader(self, data_root_path=None):
        """Create calibration dataloader."""
        dataloader = self.create_dataloader(data_root_path=data_root_path)
        return default_calibration_dataloader(dataloader)

    def get_first_batch(self, dataloader=None, data_root_path=None):
        """Get first batch of dataloader."""
        dataloader = dataloader or self.create_dataloader(data_root_path=data_root_path)
        return next(iter(dataloader))

    def update_component(self):
        return None
