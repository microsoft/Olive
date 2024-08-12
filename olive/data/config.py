# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from olive.common.config_utils import ConfigBase, NestedConfig, validate_lowercase
from olive.common.import_lib import import_user_module
from olive.common.pydantic_v1 import Field, root_validator, validator
from olive.data.constants import DataComponentType, DefaultDataComponent, DefaultDataContainer
from olive.data.registry import Registry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from olive.data.container.data_container import DataContainer


class DataComponentConfig(NestedConfig):
    _nested_field_name = "params"

    type: str = None
    params: Dict = Field(default_factory=dict)

    @validator("type", pre=True)
    def validate_type(cls, v):
        return validate_lowercase(v)


DefaultDataComponentCombos = {
    DataComponentType.LOAD_DATASET.value: DefaultDataComponent.LOAD_DATASET.value,
    DataComponentType.PRE_PROCESS_DATA.value: DefaultDataComponent.PRE_PROCESS_DATA.value,
    DataComponentType.POST_PROCESS_DATA.value: DefaultDataComponent.POST_PROCESS_DATA.value,
    DataComponentType.DATALOADER.value: DefaultDataComponent.DATALOADER.value,
}


class DataConfig(ConfigBase):
    name: str
    type: str = DefaultDataContainer.DATA_CONTAINER.value

    # user script to define and register the components
    user_script: Union[Path, str] = None
    script_dir: Union[Path, str] = None

    load_dataset_config: DataComponentConfig = None
    pre_process_data_config: DataComponentConfig = None
    post_process_data_config: DataComponentConfig = None
    dataloader_config: DataComponentConfig = None

    @root_validator(pre=True)
    def validate_data_config(cls, values):
        if values.get("user_script"):
            import_user_module(values["user_script"], values.get("script_dir"))
        return values

    @validator("name", pre=True)
    def validate_name(cls, v):
        pattern = r"^[A-Za-z0-9_]+$"
        if not re.match(pattern, v):
            raise ValueError(f"DataConfig name {v} should only contain letters, numbers and underscore.")
        return v

    @validator("type", pre=True)
    def validate_type(cls, v):
        if v is not None and Registry.get_container(v) is None:
            raise ValueError(f"Invalid/unknown DataConfig type: {v}")
        return validate_lowercase(v)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_components()
        self._fill_in_params()

    def _update_components(self):
        """Update the components in the data config with default_components if user do not provide."""
        default_components = self._get_default_components()

        # update components from default_components
        components = self.components
        for k, v in default_components.items():
            # do deepcopy here since we don't want to update the default_components
            if components[k]:
                # type is string, so we don't need to deepcopy
                components[k].type = components[k].type or v.type
                # v.params is a dict, so we deepcopy it
                components[k].params = components[k].params or deepcopy(v.params)
            else:
                # v is a DataComponentConfig object, so we deepcopy it
                components[k] = deepcopy(v)

        self.load_dataset_config = components[DataComponentType.LOAD_DATASET.value]
        self.pre_process_data_config = components[DataComponentType.PRE_PROCESS_DATA.value]
        self.post_process_data_config = components[DataComponentType.POST_PROCESS_DATA.value]
        self.dataloader_config = components[DataComponentType.DATALOADER.value]

    def _get_default_components(self):
        """Resolve the default component type."""
        # 1. get default_components_type from DataContainer or DefaultDataComponentCombos
        dc_cls = Registry.get_container(self.type)
        # deepcopy dc_cls.default_components_type since we don't want to update dc_cls.default_components_type
        default_components_type = deepcopy(dc_cls.default_components_type) or {}
        # update default_components_type with task_type for huggingface case
        self._update_default_component_type_with_task_type(dc_cls, default_components_type)
        # update default_components_type with DefaultDataComponentCombos
        # for those components not defined in the container config
        for k, v in DefaultDataComponentCombos.items():
            if k not in default_components_type:
                default_components_type[k] = v

        # 2. get default_components from default_components_type
        return {k: DataComponentConfig(type=v) for k, v in default_components_type.items()}

    def _fill_in_params(self):
        """Fill in the default parameters for each component.

        For each component, loop over the parameters from the function signature of the component
        and fill in the default in params if the key doesn't already exist.

        Priority is: component.params > default value
        """
        from inspect import signature

        for k, v in self.components.items():
            # Ignore to expose the first argument for pre_process, dataloader and post_process
            # since the first argument is dataset which is generated by load_dataset
            if_ignore_first_arg = k != DataComponentType.LOAD_DATASET.value
            component = Registry.get_component(k, v.type)

            params = signature(component).parameters
            for idx, (param, info) in enumerate(params.items()):
                # skip the first argument for pre_process, dataloader and post_process
                if if_ignore_first_arg and idx == 0:
                    continue

                # Use the default value from the function signature
                if param not in v.params:
                    if info.kind in (info.VAR_POSITIONAL, info.VAR_KEYWORD):
                        pass
                    elif info.default is info.empty:
                        logger.debug(
                            "Missing parameter %s for component %s with type %s. Set to None.", param, k, v.type
                        )
                        v.params[param] = None
                    else:
                        v.params[param] = params[param].default

    @property
    def components(self):
        return {
            DataComponentType.LOAD_DATASET.value: self.load_dataset_config,
            DataComponentType.PRE_PROCESS_DATA.value: self.pre_process_data_config,
            DataComponentType.POST_PROCESS_DATA.value: self.post_process_data_config,
            DataComponentType.DATALOADER.value: self.dataloader_config,
        }

    @property
    def load_dataset(self):
        """Get the dataset from data config."""
        return Registry.get_load_dataset_component(self.load_dataset_config.type)

    @property
    def pre_process(self):
        """Get the pre-process from data config."""
        return Registry.get_pre_process_component(self.pre_process_data_config.type)

    @property
    def post_process(self):
        """Get the post-process from data config."""
        return Registry.get_post_process_component(self.post_process_data_config.type)

    @property
    def dataloader(self):
        """Get the dataloader from data config."""
        return Registry.get_dataloader_component(self.dataloader_config.type)

    @property
    def load_dataset_params(self):
        """Get the parameters from dataset."""
        return self.load_dataset_config.params

    @property
    def pre_process_params(self):
        """Get the parameters from pre-process."""
        return self.pre_process_data_config.params

    @property
    def post_process_params(self):
        """Get the parameters from post-process."""
        return self.post_process_data_config.params

    @property
    def dataloader_params(self):
        """Get the parameters from dataloader."""
        return self.dataloader_config.params

    def to_data_container(self) -> "DataContainer":
        """Convert the data config to the data container."""
        dc_cls = Registry.get_container(self.type)
        return dc_cls(config=self)

    def _update_default_component_type_with_task_type(self, dc_cls, default_components_type):
        for component_name, config in self.components.items():
            if config and config.params:
                task_type = config.params.get("task")
                if task_type:
                    task_specific_override = dc_cls.task_type_components_map.get(
                        task_type.replace("-with-past", ""), {}
                    ).get(component_name)
                    if task_specific_override:
                        default_components_type[component_name] = task_specific_override
