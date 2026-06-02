# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator

from olive.common.config_utils import NestedConfig
from olive.common.constants import LOCAL_INPUT_MODEL_ID
from olive.common.utils import hash_dict, hash_file, hash_string
from olive.model.config.registry import get_model_handler, is_valid_model_type
from olive.resource_path import create_resource_path

logger = logging.getLogger(__name__)


class ModelConfig(NestedConfig):
    """Input model config which will be used to create the model handler."""

    type: str = Field(description="The type of the model handler.")
    config: dict = Field(description="The config for the model handler. Used to initialize the model handler.")

    @field_validator("type")
    @classmethod
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

    def get_components(self) -> Optional[list[str]]:
        """Return the list of component names exposed by this input model, or None if single-component.

        * ``CompositeModel`` -> the configured ``model_component_names`` list.
        * ``DiffusersModel`` -> the variant-specific exportable components (via the handler).
        * Anything else -> ``None`` (single-component model).
        """
        model_type = self.type
        if model_type == "compositemodel":
            return list(self.config.get("model_component_names") or [])
        if model_type == "diffusersmodel":
            handler = self.create_model()
            return [str(c) for c in handler.get_exportable_components()]
        return None

    def select_components(self, names: list[str]) -> "ModelConfig":
        """Return a new ModelConfig holding only the named components of a CompositeModel.

        Returns the unwrapped child component ``ModelConfig`` when exactly one name is given;
        returns a new ``CompositeModel`` ``ModelConfig`` containing the subset (in the requested
        order) otherwise. Raises ``ValueError`` if invoked on a non-composite model or if any
        name is missing from ``model_component_names``.
        """
        if self.type != "compositemodel":
            raise ValueError(
                f"select_components is only supported on CompositeModel input configs (got type {self.type!r})."
            )
        if not names:
            raise ValueError("select_components requires a non-empty list of names.")
        component_names = self.config.get("model_component_names") or []
        model_components = self.config.get("model_components") or []
        if len(component_names) != len(model_components):
            raise ValueError("CompositeModel config has mismatched model_components and model_component_names lengths.")
        missing = [n for n in names if n not in component_names]
        if missing:
            raise ValueError(f"Unknown component name(s) {missing}. Available components: {list(component_names)}.")
        component_map = dict(zip(component_names, model_components))
        selected = [deepcopy(component_map[n]) for n in names]
        if len(selected) == 1:
            child = selected[0]
            if isinstance(child, ModelConfig):
                return child
            return ModelConfig.model_validate(child)
        new_config = {
            **{k: v for k, v in self.config.items() if k not in ("model_components", "model_component_names")},
            "model_components": selected,
            "model_component_names": list(names),
        }
        return ModelConfig(type=self.type, config=new_config)

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
        return hash_dict({"model_identifier": model_identifier, "model_config": model_config.model_dump()})[:8]

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

    def _get_model_files_hash(self, config: dict):
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
        else:
            # some model path or adapter path may be a hf repo string.
            # if it is not a file or a directory, hash the path string.
            file_hashes.append(hash_string(file_path.as_posix()))
        return file_hashes
