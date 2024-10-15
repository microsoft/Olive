# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict

from olive.common.config_utils import NestedConfig
from olive.common.constants import LOCAL_INPUT_MODEL_ID
from olive.common.pydantic_v1 import Field, validator
from olive.common.utils import hash_dict, hash_file, hash_string
from olive.model.config.registry import get_model_handler, is_valid_model_type
from olive.resource_path import create_resource_path

logger = logging.getLogger(__name__)


class ModelConfig(NestedConfig):
    """Input model config which will be used to create the model handler."""

    type: str = Field(description="The type of the model handler.")
    config: dict = Field(description="The config for the model handler. Used to initialize the model handler.")

    @validator("type")
    def validate_type(cls, v):
        if not is_valid_model_type(v):
            raise ValueError(f"Unknown model type {v}")
        return v.lower()

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

    def get_model_id(self):
        for v in self.config.values():
            if callable(v):
                return LOCAL_INPUT_MODEL_ID

        model_identifier = self.get_model_identifier()
        model_config = deepcopy(self)
        model_config.config.pop("model_path", None)
        model_config.config.pop("adapter_path", None)
        if model_config.config.get("model_attributes"):
            model_config.config["model_attributes"].pop("additional_files", None)
            model_config.config["model_attributes"].pop("_name_or_path", None)
        return hash_dict({"model_identifier": model_identifier, "model_config": model_config.dict()})[:8]

    def get_model_identifier(self):
        model_path = self.config.get("model_path")
        if model_path:
            model_path_resource_path = create_resource_path(model_path)
            if (
                self.type == "hfmodel"
                and model_path_resource_path.is_string_name()
                and self.config.get("adapter_path") is None
            ):
                try:
                    # huggingface_hub is a dependency of transformers
                    from huggingface_hub import repo_info
                except ImportError as exc:
                    logger.exception(
                        "huggingface_hub is not installed. "
                        "Please install huggingface_hub for supporting Huggingface model."
                    )
                    raise ImportError("huggingface_hub is not installed.") from exc
                return repo_info(model_path).sha

            if model_path_resource_path.is_azureml_resource():
                return model_path_resource_path.get_path()

        file_hashes = self._get_model_files_hash(self.config)
        sorted_file_hashes = sorted(file_hashes)
        return hash_string("".join(sorted_file_hashes))

    def _get_model_files_hash(self, config: Dict):
        keys = ["model_path", "adapter_path", "model_script", "script_dir"]
        local_resource_paths = [Path(config[key]) for key in keys if config.get(key)]

        additional_files = (config.get("model_attributes") or {}).get("additional_files") or []
        local_resource_paths.extend(Path(f) for f in additional_files)
        file_hashes = []
        for local_resource_path in local_resource_paths:
            file_hashes.extend(self._get_file_hash(local_resource_path))
        return file_hashes

    def _get_file_hash(self, file_path: Path):
        file_hashes = []
        if file_path.is_file():
            file_hashes.append(hash_file(file_path, block_size=1024 * 1024)[:8])
        elif file_path.is_dir():
            for file in file_path.iterdir():
                file_hashes.extend(self._get_file_hash(file))
        return file_hashes
