# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.config_utils import ConfigBase
from olive.data_container.constants import DefaultDataContainer
from olive.data_container.registry import Registry


class DataComponentConfig(ConfigBase):
    name: str = None
    type: str = None
    params: dict = None


class DataContainerConfig(ConfigBase):
    name: str = DefaultDataContainer.DATA_CONTAINER.value
    type: str = DefaultDataContainer.DATA_CONTAINER.value
    components: dict = {
        # the empty component will be overwrote by the components in the given data container
        # priority: config data components > default data components
        "dataset": DataComponentConfig(),
        "pre_process": DataComponentConfig(),
        "post_process": DataComponentConfig(),
        "dataloader": DataComponentConfig(),
    }

    def get_container_cls(self):
        return Registry.get_container(self.type)

    def create_container(self):
        container_cls = self.get_container_cls()
        return container_cls(config=self)
