# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator


class DispatcherType(str, Enum):
    Remote = "Remote"


class Dispatcher(ABC):
    dispatcher_type: DispatcherType

    @abstractmethod
    def load_config(self, config_path: str):
        raise NotImplementedError


_type_to_dispatcher_path = {
    DispatcherType.Remote: "olive.workflows.dispatcher.remote_dispatcher.RemoteDispatcher",
}


def import_dispatcher_from_type(dispatcher_type: DispatcherType):
    dispatcher_path = _type_to_dispatcher_path[dispatcher_type]
    module_path, class_name = dispatcher_path.rsplit(".", 1)
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
        dispatcher_class = import_dispatcher_from_type(self.type)
        return dispatcher_class(self.config_path)
