# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import ABC
from enum import Enum
import importlib
from pathlib import Path

from olive.common.pydantic_v1 import validator

from olive.common.config_utils import ConfigBase

class DispatcherType(str, Enum):
    Remote = "Remote"


class Dispatcher(ABC):
    dispatcher_type: DispatcherType
    
    def __init__(self, dispatcher_config: dict):
        self.dispatcher_config = dispatcher_config


_type_to_dispactcher_path = {
    DispatcherType.Remote: "olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher",
}

def import_dispactcher_from_type(dispactcher_type: DispatcherType):
    dispactcher_path = _type_to_dispactcher_path[dispactcher_type]
    module_path, class_name = dispactcher_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class RunDispatcherConfig(ConfigBase):
    type: DispatcherType
    config_path: str
    
    @validator("config_path")
    def validate_config_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Config file {v} does not exist.")
        return v

    def create_dispatcher(self):
        dispactcher_class = import_dispactcher_from_type(self.type)
        return dispactcher_class(self.config_path)
