# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.config_utils import NestedConfig
from olive.common.pydantic_v1 import Field, validator
from olive.model.config.registry import get_model_handler, is_valid_model_type
from olive.resource_path import create_resource_path


class ModelConfig(NestedConfig):
    """Input model config which will be used to create the model handler."""

    type: str = Field(description="The type of the model handler.")
    config: dict = Field(description="The config for the model handler. Used to initialize the model handler.")

    @validator("type")
    def validate_type(cls, v):
        if not is_valid_model_type(v):
            raise ValueError(f"Unknown model type {v}")
        return v

    def get_resource_strings(self):
        cls = get_model_handler(self.type)
        resource_keys = cls.get_resource_keys()
        return {k: v for k, v in self.config.items() if k in resource_keys}

    def get_resource_paths(self):
        resources = self.get_resource_strings()
        return {k: create_resource_path(v) for k, v in resources.items()}

    def create_model(self):
        cls = get_model_handler(self.type)
        return cls(**self.config)
