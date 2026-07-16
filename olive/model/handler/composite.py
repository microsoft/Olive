# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Optional, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.utils import dict_diff
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

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

        # When components are not provided but model_path is a directory of per-component ONNX
        # subfolders, discover them using the subfolder names as component names.
        if model_components is None:
            discovered = self._discover_components(model_path)
            if not discovered:
                raise ValueError(
                    "CompositeModelHandler requires model_components, or a model_path directory containing "
                    "per-component ONNX subfolders."
                )
            model_components, model_component_names = discovered

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
        self.model_component_names = model_component_names

    @staticmethod
    def _discover_components(
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
    ) -> Optional[tuple[list[dict[str, Any]], list[str]]]:
        """Build component configs from a directory of per-component ONNX subfolders.

        Returns ``(model_components, model_component_names)`` or ``None`` if discovery is not
        applicable (model_path is not a local directory of component subfolders).
        """
        from olive.model.utils.onnx_utils import discover_onnx_components

        if not model_path or not Path(str(model_path)).is_dir():
            return None
        discovered = discover_onnx_components(str(model_path))
        if not discovered:
            return None
        names = [name for name, _ in discovered]
        components = [
            {"type": "ONNXModel", "config": {"model_path": str(model_path), "onnx_file_name": onnx_rel}}
            for _, onnx_rel in discovered
        ]
        return components, names

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
        missing = [n for n in names if n not in self.model_component_names]
        if missing:
            raise ValueError(
                f"Unknown component name(s) {missing}. Available components: {list(self.model_component_names)}."
            )
        component_map = dict(zip(self.model_component_names, self._model_components))
        selected = [component_map[n] for n in names]
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
