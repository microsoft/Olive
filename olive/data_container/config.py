# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.common.config_utils import ConfigBase
from olive.data_container.constants import DataComponentType, DefaultDataComponent, DefaultDataContainer
from olive.data_container.registry import Registry

logger = logging.getLogger(__name__)


class DataComponentConfig(ConfigBase):
    name: str = None
    type: str = None
    params: dict = None


DefaultDataComponentCombos = {
    DataComponentType.DATASET.value: DefaultDataComponent.DATASET.value,
    DataComponentType.PRE_PROCESS.value: DefaultDataComponent.PRE_PROCESS.value,
    DataComponentType.POST_PROCESS.value: DefaultDataComponent.POST_PROCESS.value,
    DataComponentType.DATALOADER.value: DefaultDataComponent.DATALOADER.value,
}


class DataContainerConfig(ConfigBase):
    name: str = DefaultDataContainer.DATA_CONTAINER.value
    type: str = DefaultDataContainer.DATA_CONTAINER.value

    # used to store the params for each component
    params_config: dict = None

    # use to update default components
    # 1. update default_components_type from BaseContainer or DefaultDataComponentCombos
    # 2. update default_components from default_components_type
    # 3. update components from default_components
    components: dict[str, DataComponentConfig] = None
    default_components: dict[str, DataComponentConfig] = None
    default_components_type: dict[str, str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_components()
        self.fill_in_params()

    def update_components(self):
        """
        Update the components in the data container with default_components if user do not provide.
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
        dc_cls = Registry.get_container(self.type)
        return dc_cls(config=self)

    def _update_default_component_type_with_task_type(self, dc_cls):
        if not self.params_config:
            return
        task_type = self.params_config.get("task_type", None)
        if task_type:
            self.default_components_type.update(dc_cls.task_type_components_map.get(task_type, {}))
