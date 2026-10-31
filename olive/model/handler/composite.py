# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.utils import dict_diff
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import LOCAL_RESOURCE_TYPES, OLIVE_RESOURCE_ANNOTATIONS, ResourceType

logger = logging.getLogger(__name__)


@model_handler_registry("CompositeModel")
class CompositeModelHandler(OliveModelHandler):
    """CompositeModel represents multiple component models.

    The only responsibility of CompositeModelHandler is to provider a get_model_components which will iterate all the
    child models.

    CompositeModelHandler is a collection of Models. All the child model in the container should have same model type.
    """

    resource_keys: tuple[str, ...] = ("model_path",)
    json_config_keys: tuple[str, ...] = ("model_component_names",)

    def __init__(
        self,
        model_components: Optional[list[Union[OliveModelHandler, dict[str, Any]]]] = None,
        model_component_names: Optional[list[str]] = None,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.COMPOSITE_MODEL,
            model_attributes=model_attributes,
        )

        # Load the complete package before falling back to ONNX subfolder discovery.
        if model_components is None:
            discovered = self._discover_components(model_path)
            if not discovered:
                raise ValueError(
                    "CompositeModelHandler requires model_components, or a model_path directory containing "
                    "per-component ONNX subfolders."
                )
            model_components = discovered["model_components"]
            model_component_names = discovered["model_component_names"]
            self.model_attributes = {
                **(discovered.get("model_attributes") or {}),
                **(self.model_attributes or {}),
            }
            additional_files = self.model_attributes.get("additional_files") or []
            if additional_files:
                for component in model_components:
                    attributes = component["config"].get("model_attributes") or {}
                    component["config"]["model_attributes"] = attributes
                    files_by_name = {}
                    for path in [*additional_files, *(attributes.get("additional_files") or [])]:
                        files_by_name[Path(path).name] = path
                    attributes["additional_files"] = list(files_by_name.values())

        if model_component_names is None:
            raise ValueError("CompositeModelHandler requires model_component_names when model_components is provided.")

        self._model_components = [
            validate_config(m, ModelConfig).create_model() if isinstance(m, dict) else m for m in model_components
        ]
        if not all(isinstance(m, OliveModelHandler) for m in self._model_components):
            raise ValueError("All components must be OliveModelHandler or dict.")

        if len(self._model_components) != len(model_component_names):
            raise ValueError(
                f"Number of components ({len(self._model_components)}) and names "
                f"({len(model_component_names)}) must match."
            )
        if len(set(model_component_names)) != len(model_component_names):
            raise ValueError("Composite model component names must be unique.")
        self.model_component_names = model_component_names

    @staticmethod
    def _discover_components(
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
    ) -> Optional[dict[str, Any]]:
        """Load a model package config, or discover per-component ONNX subfolders.

        Returns the composite config or ``None`` when model_path is not a component package.
        """
        from olive.model.utils.onnx_utils import discover_onnx_components

        if not model_path or not Path(str(model_path)).is_dir():
            return None
        package_dir = Path(str(model_path))
        config_path = package_dir / "model_config.json"
        if config_path.is_file():
            package = ModelConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            if package.type != "compositemodel":
                raise ValueError(f"Expected a CompositeModel package config in {config_path}, got {package.type!r}.")
            source_root = package.config.get("model_path")
            if not source_root:
                components = package.config.get("model_components") or []
                source_root = components[0]["config"].get("model_path") if components else None
                if not source_root or any(
                    component["type"].lower() != "onnxmodel"
                    or not component["config"].get("onnx_file_name")
                    or component["config"].get("model_path") != source_root
                    for component in components
                ):
                    raise ValueError(
                        f"CompositeModel package config in {config_path} requires model_path unless all ONNX "
                        "components share a directory and define onnx_file_name."
                    )
            if isinstance(source_root, dict):
                resource_type = ResourceType(source_root.get("type"))
                if resource_type not in LOCAL_RESOURCE_TYPES:
                    raise ValueError(f"CompositeModel package root in {config_path} must be a local path.")
                source_root = source_root["config"]["path"]
            source_root = str(source_root).replace("\\", "/").rstrip("/")
            rebased = CompositeModelHandler._rebase_package_paths(package.config, source_root, package_dir.resolve())
            rebased["model_path"] = str(package_dir.resolve())
            return rebased
        discovered = discover_onnx_components(str(model_path))
        if not discovered:
            return None
        names = [name for name, _ in discovered]
        components = [
            {"type": "ONNXModel", "config": {"model_path": str(model_path), "onnx_file_name": onnx_rel}}
            for _, onnx_rel in discovered
        ]
        return {"model_components": components, "model_component_names": names}

    @staticmethod
    def _rebase_package_paths(
        config: dict[str, Any],
        source_root: str,
        destination_root: Path,
        model_type: str = "CompositeModel",
    ) -> dict[str, Any]:
        def rebase_path(value: str) -> str:
            path = value.replace("\\", "/")
            if path == source_root:
                return str(destination_root)
            if path.startswith(f"{source_root}/"):
                return str(destination_root / PurePosixPath(path[len(source_root) + 1 :]))
            return value

        def rebase_resource(value: Any) -> Any:
            if isinstance(value, dict) and ResourceType(value.get("type")) in LOCAL_RESOURCE_TYPES:
                value["config"]["path"] = rebase_path(value["config"]["path"])
            elif isinstance(value, str):
                value = rebase_path(value)
            return value

        model = ModelConfig(type=model_type, config=deepcopy(config))
        for key, resource in model.get_resource_strings().items():
            model.config[key] = rebase_resource(resource)
        attributes = model.config.get("model_attributes") or {}
        if attributes.get("additional_files"):
            attributes["additional_files"] = [rebase_resource(path) for path in attributes["additional_files"]]
        for component in model.config.get("model_components") or []:
            component["config"] = CompositeModelHandler._rebase_package_paths(
                component["config"], source_root, destination_root, component["type"]
            )
        return model.config

    @property
    def model_components(self):
        for m in self._model_components:
            # the parent attributes should be inherited by the child model
            # child attributes take precedence
            m.model_attributes = {**(self.model_attributes or {}), **(m.model_attributes or {})}
            yield m

    def to_json(self, check_object: bool = False):
        json_dict = super().to_json(check_object)
        json_dict["config"]["model_components"] = []
        for m in self._model_components:
            component_json = m.to_json(check_object)
            # only keep attributes that are different from the parent
            component_json["config"]["model_attributes"] = dict_diff(
                component_json["config"]["model_attributes"], self.model_attributes
            )
            json_dict["config"]["model_components"].append(component_json)

        return serialize_to_json(json_dict, check_object)

    def get_model_components(self) -> list[tuple[str, OliveModelHandler]]:
        return zip(self.model_component_names, self.model_components)

    def select_components(self, names: list[str]) -> "OliveModelHandler":
        """Return a handler holding only the named components.

        Returns the unwrapped child handler when exactly one name is given; returns a new
        ``CompositeModelHandler`` containing the subset (in the requested order) otherwise.
        Raises ``ValueError`` if any name is missing from ``model_component_names``.
        """
        if not names:
            raise ValueError("select_components requires a non-empty list of names.")
        if len(set(names)) != len(names):
            raise ValueError("select_components requires unique component names.")
        missing = [n for n in names if n not in self.model_component_names]
        if missing:
            raise ValueError(
                f"Unknown component name(s) {missing}. Available components: {list(self.model_component_names)}."
            )
        component_map = dict(zip(self.model_component_names, self._model_components))
        selected = [deepcopy(component_map[n]) for n in names]
        if len(selected) == 1:
            child = selected[0]
            child.model_attributes = {**(self.model_attributes or {}), **(child.model_attributes or {})}
            return child
        return CompositeModelHandler(
            model_components=selected,
            model_component_names=list(names),
            model_path=self.model_path,
            model_attributes=self.model_attributes,
        )

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError

    @property
    def size_on_disk(self) -> int:
        """Compute size of the model on disk."""
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
        rank: Optional[int] = None,
    ):
        raise RuntimeError("CompositeModelHandler doesn't have a session of its own")

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        raise RuntimeError("CompositeModelHandler doesn't have a session of its own")
