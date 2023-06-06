# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pydantic import validator

from olive.common.config_utils import ConfigBase
from olive.model.model_config import IOConfig


class HFComponent(ConfigBase):
    name: str
    io_config: Optional[Union[IOConfig, str, Dict[str, Any]]] = None
    component_func: Optional[Union[str, Callable]] = None
    dummy_inputs_func: Optional[Union[str, Callable]] = None


class HFConfig(ConfigBase):
    model_name: str = None
    task: str = None
    feature: str = "default"
    # TODO: remove model_class and only use task
    model_class: str = None
    model_type: str = None
    state_dict_path: Union[str, Path] = None
    use_custom_implementation: bool = False
    components: List[HFComponent] = None
    config: Dict[str, Any] = None
    dataset: Dict[str, Any] = None

    @validator("model_class", always=True)
    def task_or_model_class_required(cls, v, values):
        if values["model_name"]:
            if not v and not values.get("task", None):
                raise ValueError("Either task or model_class must be specified")
            return v

    @validator("components")
    def validate_components(cls, v, values):
        if values["use_custom_implementation"]:
            for component in v:
                if component.io_config is None:
                    raise ValueError("io_config must be specified for all components")
                if component.component_func is None:
                    raise ValueError("component_func must be specified for all components")
                if component.dummy_inputs_func is None:
                    raise ValueError("dummy_inputs_func must be specified for all components")
        return v


class HFModelBase(torch.nn.Module):
    """Base class for all HF models"""

    def __init__(
        self,
        components: Dict[str, Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        for component in components:
            setattr(self, component.name, component.component_func)
        self.config = config
