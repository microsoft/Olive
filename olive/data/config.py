# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from olive.common.config_utils import ConfigBase
from olive.common.import_lib import import_user_module
from olive.common.pydantic_v1 import validator
from olive.data.constants import DataComponentType, DefaultDataComponent, DefaultDataContainer
from olive.data.registry import Registry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from olive.data.container.data_container import DataContainer


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

    # user script to define and register the components
    user_script: Union[Path, str] = None
    script_dir: Union[Path, str] = None

    # use to update default components
    # 1. update default_components_type from DataContainer or DefaultDataComponentCombos
    # 2. update default_components from default_components_type
    # 3. update components from default_components
    components: Dict[str, DataComponentConfig] = None

    # Customer should not update this field !!
    # default_components is used to store mapping from default component name to component config
    # which will be updated by default_components_type automatically for different subclass of
    # DataContainer like: HuggingfaceContainer, DummyDataContainer
    # key: is the value from DataComponentType
    # value: is corresponding component config
    default_components: Dict[str, DataComponentConfig] = None

    # Customer should not update this field !!
    # default_components_type is used to store mapping from default component name to component type
    # key: is the value from DataComponentType
    # value: is corresponding component function name registered in Registry
    # for example, {DataComponentType.LOAD_DATASET.value: "huggingface_dataset"}
    default_components_type: Dict[str, str] = None

    @validator("name", pre=True)
    def validate_name(cls, v):
        pattern = r"^[A-Za-z0-9_]+$"
        if not re.match(pattern, v):
            raise ValueError(f"DataConfig name {v} should only contain letters, numbers and underscore.")
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # call import_user_module to load the user script once and register the components
        if self.user_script:
            import_user_module(self.user_script, self.script_dir)
        self.update_components()
        self.fill_in_params()

    def update_components(self):
        """Update the components in the data config with default_components if user do not provide."""
        self.components = self.components or {}
        self.default_components = self.default_components or {}
        self._update_default_component_type()
        self._update_default_component()
        for k, v in self.default_components.items():
            # do deepcopy here since we don't want to update the default_components
            if k not in self.components:
                # v is a DataComponentConfig object, so we deepcopy it
                self.components[k] = deepcopy(v)
            else:
                # both are strings, so we don't need to deepcopy
                self.components[k].type = self.components[k].type or v.type
                self.components[k].name = self.components[k].name or v.name
                # v.params is a dict, so we deepcopy it
                self.components[k].params = self.components[k].params or deepcopy(v.params)

    def _update_default_component_type(self):
        """Resolve the default component type."""
        dc_cls = Registry.get_container(self.type)
        # deepcopy dc_cls.default_components_type since we don't want to update dc_cls.default_components_type
        self.default_components_type = deepcopy(dc_cls.default_components_type) or {}
        # 1. update default_components_type with task_type for huggingface case
        self._update_default_component_type_with_task_type(dc_cls)
        # 2. update default_components_type with DefaultDataComponentCombos
        # for those components not defined in the container config
        for k, v in DefaultDataComponentCombos.items():
            if k not in self.default_components_type:
                self.default_components_type[k] = v

    def _update_default_component(self):
        """Resolve the default component type."""
        for k, v in self.default_components_type.items():
            self.default_components[k] = DataComponentConfig(type=v, name=v, params={})

    def fill_in_params(self):
        """Fill in the default parameters for each component.

        For each component, we will do the following steps:
            1. If params_config["component_kwargs"] is not None, use the params_config["component_kwargs"]
            to update component.params. Higher priority than the following steps.
        Loop over the parameters from the function signature of the component:
            2. If defined in params_config, use it to fill in the params
            3. Else if already defined in component.params, use it directly
            4. Else Use the default value from the function signature

        Priority is: component_kwargs > params_config > component.params > default value
        """
        from inspect import signature

        self.params_config = deepcopy(self.params_config) or {}
        component_kwargs = self.params_config.pop("component_kwargs", {})

        for k, v in self.components.items():
            # Ignore to expose the first argument for pre_process, dataloader and post_process
            # since the first argument is dataset which is generated by load_dataset
            if_ignore_first_arg = k != DataComponentType.LOAD_DATASET.value
            component = Registry.get_component(k, v.type)
            # 1. use the params_config["component_kwargs"] to update component.params
            v.params.update(component_kwargs.get(k, {}))
            params = signature(component).parameters
            for idx, (param, info) in enumerate(params.items()):
                # skip the first argument for pre_process, dataloader and post_process
                if if_ignore_first_arg and idx == 0:
                    continue
                # skip the params already defined in component_kwargs
                if param in component_kwargs.get(k, {}):
                    continue
                # 2. Update the param using params_config
                if param in self.params_config:
                    v.params[param] = self.params_config[param]
                    continue
                # 3. Use value from component.params if already defined
                # 4. Use the default value from the function signature
                if param not in v.params:
                    if info.kind in (info.VAR_POSITIONAL, info.VAR_KEYWORD):
                        continue
                    elif info.default is info.empty:
                        logger.debug(
                            "Missing parameter %s for component %s with type %s. Set to None.", param, k, v.type
                        )
                        v.params[param] = None
                    else:
                        v.params[param] = params[param].default

    def get_components_params(self):
        """Get the parameters from data config."""
        return {k: v.params for k, v in self.components.items()}

    @property
    def load_dataset(self):
        """Get the dataset from data config."""
        name = self.components[DataComponentType.LOAD_DATASET.value].type or DefaultDataComponent.LOAD_DATASET.value
        return Registry.get_load_dataset_component(name)

    @property
    def pre_process(self):
        """Get the pre-process from data config."""
        name = (
            self.components[DataComponentType.PRE_PROCESS_DATA.value].type
            or DefaultDataComponent.PRE_PROCESS_DATA.value
        )
        return Registry.get_pre_process_component(name)

    @property
    def post_process(self):
        """Get the post-process from data config."""
        name = (
            self.components[DataComponentType.POST_PROCESS_DATA.value].type
            or DefaultDataComponent.POST_PROCESS_DATA.value
        )
        return Registry.get_post_process_component(name)

    @property
    def dataloader(self):
        """Get the dataloader from data config."""
        name = self.components[DataComponentType.DATALOADER.value].type or DefaultDataComponent.DATALOADER.value
        return Registry.get_dataloader_component(name)

    @property
    def load_dataset_params(self):
        """Get the parameters from dataset."""
        return self.components[DataComponentType.LOAD_DATASET.value].params

    @property
    def pre_process_params(self):
        """Get the parameters from pre-process."""
        return self.components[DataComponentType.PRE_PROCESS_DATA.value].params

    @property
    def post_process_params(self):
        """Get the parameters from post-process."""
        return self.components[DataComponentType.POST_PROCESS_DATA.value].params

    @property
    def dataloader_params(self):
        """Get the parameters from dataloader."""
        return self.components[DataComponentType.DATALOADER.value].params

    def to_data_container(self) -> "DataContainer":
        """Convert the data config to the data container."""
        dc_cls = Registry.get_container(self.type)
        return dc_cls(config=self)

    def _update_default_component_type_with_task_type(self, dc_cls):
        if not self.params_config:
            return
        task_type = self.params_config.get("task", None)
        if task_type:
            self.default_components_type.update(dc_cls.task_type_components_map.get(task_type, {}))
