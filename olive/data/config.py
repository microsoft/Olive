# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Dict

from olive.common.config_utils import ConfigBase
from olive.data.constants import DataComponentType, DefaultDataComponent, DefaultDataContainer
from olive.data.registry import Registry

logger = logging.getLogger(__name__)


class DataComponentConfig(ConfigBase):
    name: str = None
    type: str = None
    params: Dict = None


DefaultDataComponentCombos = {
    DataComponentType.LOAD_DATASET.value: DefaultDataComponent.LOAD_DATASET.value,
    DataComponentType.PRE_PROCESS_DATA.value: DefaultDataComponent.PRE_PROCESS_DATA.value,
    DataComponentType.POST_PROCESS_DATA.value: DefaultDataComponent.POST_PROCESS_DATA.value,
    DataComponentType.DATALOADER.value: DefaultDataComponent.DATALOADER.value,
}


class DataConfig(ConfigBase):
    name: str = DefaultDataContainer.DATA_CONTAINER.value
    type: str = DefaultDataContainer.DATA_CONTAINER.value

    # used to store the params for each component
    params_config: Dict = None

    # use to update default components
    # 1. update default_components_type from DataContainer or DefaultDataComponentCombos
    # 2. update default_components from default_components_type
    # 3. update components from default_components
    components: Dict[str, DataComponentConfig] = None
    default_components: Dict[str, DataComponentConfig] = None
    default_components_type: Dict[str, str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_components()
        self.fill_in_params()

    def update_components(self):
        """
        Update the components in the data config with default_components if user do not provide.
        """
        self.components = self.components or {}
        self.default_components = self.default_components or {}
        self._update_default_component_type()
        self._update_default_component()
        for k, v in self.default_components.items():
            if k not in self.components:
                self.components[k] = v
            else:
                self.components[k].type = self.components[k].type or v.type
                self.components[k].name = self.components[k].name or v.name
                self.components[k].params = self.components[k].params or v.params

    def _update_default_component_type(self):
        """
        Resolve the default component type.
        """
        dc_cls = Registry.get_container(self.type)
        self.default_components_type = dc_cls.default_components_type or {}
        # 1. update default_components_type with task_type for huggingface case
        self._update_default_component_type_with_task_type(dc_cls)
        # 2. update default_components_type with DefaultDataComponentCombos
        # for those components not defined in the container config
        for k, v in DefaultDataComponentCombos.items():
            if k not in self.default_components_type:
                self.default_components_type[k] = v

    def _update_default_component(self):
        """
        Resolve the default component type.
        """
        for k, v in self.default_components_type.items():
            self.default_components[k] = DataComponentConfig(type=v, name=v, params={})

    def fill_in_params(self):
        """
        Fill in the default parameters for each component.
        1. if prams_config is not None, use the params_config to fill in the params
        2. if params_config is None, use the default params from the function signature
        3. if there is already define params under the component, use the params directly
        """
        from inspect import signature

        self.params_config = self.params_config or {}
        for k, v in self.components.items():
            component = Registry.get_component(k, v.type)
            # 1. user function signature to fill params firstly
            params = signature(component).parameters
            for param, info in params.items():
                # 2. override the params with params_config
                if param in self.params_config:
                    v.params[param] = self.params_config[param]
                    continue
                # 3. if it already defined params under the component, use the params directly
                if param not in v.params and not param.startswith("_"):
                    if info.kind == info.VAR_POSITIONAL or info.kind == info.VAR_KEYWORD:
                        continue
                    elif info.default is info.empty:
                        logger.debug(f"Missing parameter {param} for component {k}")
                        v.params[param] = None
                    else:
                        v.params[param] = params[param].default

    def get_components_params(self):
        """
        Get the parameters from data config.
        """
        return {k: v.params for k, v in self.components.items()}

    @property
    def load_dataset(self):
        """
        Get the dataset from data config.
        """
        name = self.components[DataComponentType.LOAD_DATASET.value].type or DefaultDataComponent.LOAD_DATASET.value
        return Registry.get_load_dataset_component(name)

    @property
    def pre_process(self):
        """
        Get the pre-process from data config.
        """
        name = (
            self.components[DataComponentType.PRE_PROCESS_DATA.value].type
            or DefaultDataComponent.PRE_PROCESS_DATA.value
        )
        return Registry.get_pre_process_component(name)

    @property
    def post_process(self):
        """
        Get the post-process from data config.
        """
        name = (
            self.components[DataComponentType.POST_PROCESS_DATA.value].type
            or DefaultDataComponent.POST_PROCESS_DATA.value
        )
        return Registry.get_post_process_component(name)

    @property
    def dataloader(self):
        """
        Get the dataloader from data config.
        """
        name = self.components[DataComponentType.DATALOADER.value].type or DefaultDataComponent.DATALOADER.value
        return Registry.get_dataloader_component(name)

    @property
    def load_dataset_params(self):
        """
        Get the parameters from dataset.
        """
        return self.components[DataComponentType.LOAD_DATASET.value].params

    @property
    def pre_process_params(self):
        """
        Get the parameters from pre-process.
        """
        return self.components[DataComponentType.PRE_PROCESS_DATA.value].params

    @property
    def post_process_params(self):
        """
        Get the parameters from post-process.
        """
        return self.components[DataComponentType.POST_PROCESS_DATA.value].params

    @property
    def dataloader_params(self):
        """
        Get the parameters from dataloader.
        """
        return self.components[DataComponentType.DATALOADER.value].params

    def to_data_container(self):
        """
        Convert the data config to the data container.
        """
        dc_cls = Registry.get_container(self.type)
        return dc_cls(config=self)

    def _update_default_component_type_with_task_type(self, dc_cls):
        if not self.params_config:
            return
        task_type = self.params_config.get("task", None)
        if task_type:
            self.default_components_type.update(dc_cls.task_type_components_map.get(task_type, {}))
