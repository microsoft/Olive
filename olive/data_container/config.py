# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.config_utils import ConfigBase
from olive.data_container.constants import DefaultDataComponent, DefaultDataContainer
from olive.data_container.registry import Registry


class DataComponentConfig(ConfigBase):
    name: str = None
    type: str = None
    params: dict = None


class DataContainerConfig(ConfigBase):
    name: str = DefaultDataContainer.DATA_CONTAINER.value
    type: str = DefaultDataContainer.DATA_CONTAINER.value
    config: dict = None
    components: dict = {
        # the empty component will be overwrote by the components in the given data container
        # priority: config data components > default data components
        "dataset": DataComponentConfig(),
        "pre_process": DataComponentConfig(),
        "post_process": DataComponentConfig(),
        "dataloader": DataComponentConfig(),
    }

    def get_components_params(self):
        """
        Get the parameters of the data container.
        """
        return {k: v.params for k, v in self.components.items()}

    @property
    def dataset(self):
        """
        Get the dataset of the data container.
        """
        name = self.components["dataset"].type or DefaultDataComponent.DATASET.value
        return Registry.get_dataset_component(name)

    @property
    def pre_process(self):
        """
        Get the pre-process of the data container.
        """
        name = self.components["pre_process"].type or DefaultDataComponent.PRE_PROCESS.value
        return Registry.get_pre_process_component(name)

    @property
    def post_process(self):
        """
        Get the post-process of the data container.
        """
        name = self.components["post_process"].type or DefaultDataComponent.POST_PROCESS.value
        return Registry.get_post_process_component(name)

    @property
    def dataloader(self):
        """
        Get the dataloader of the data container.
        """
        name = self.components["dataloader"].type or DefaultDataComponent.DATALOADER.value
        return Registry.get_dataloader_component(name)

    @property
    def dataset_params(self):
        """
        Get the parameters of the dataset.
        """
        return self.components["dataset"].params

    @property
    def pre_process_params(self):
        """
        Get the parameters of the pre-process.
        """
        return self.components["pre_process"].params

    @property
    def post_process_params(self):
        """
        Get the parameters of the post-process.
        """
        return self.components["post_process"].params

    @property
    def dataloader_params(self):
        """
        Get the parameters of the dataloader.
        """
        return self.components["dataloader"].params

    def to_data_container(self):
        """
        Convert the data container config to the data container.
        """
        dc_cls = Registry.get_data_container(self.type)
        return dc_cls(config=self)
