# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator
from olive.model.config.registry import get_model_handler, is_valid_model_type
from olive.resource_path import create_resource_path


class ModelConfig(ConfigBase):
    type: str  # noqa: A003
    config: dict

    @validator("type")
    def validate_type(cls, v):
        if not is_valid_model_type(v):
            raise ValueError(f"Unknown model type {v}")
        return v

    def get_resource_keys(self):
        cls = get_model_handler(self.type)
        return cls.resource_keys

    def get_resource_paths(self):
        resource_keys = self.get_resource_keys()
        return {k: create_resource_path(v) for k, v in self.config.items() if k in resource_keys}

    def create_model(self):
        cls = get_model_handler(self.type)
        return cls(**self.config)
