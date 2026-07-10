# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Wrap a multi-component CompositeModel ORT-GenAI package as a single ONNXModel handler."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from olive.common.utils import hardlink_copy_dir
from olive.model import ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec

logger = logging.getLogger(__name__)


class CompositeToOnnxPackage(Pass):
    """Repackage a CompositeModel ORT-GenAI package as a single :class:`ONNXModelHandler`.

    ``MobiusBuilder`` and similar passes emit multi-component ORT-GenAI packages as a
    :class:`CompositeModelHandler` whose components live in subdirectories::

        output_dir/
          genai_config.json
          tokenizer.json
          decoder/model.onnx
          vision_encoder/model.onnx
          audio_encoder/model.onnx
          embedding/model.onnx

    Two consumers need a concrete :class:`ONNXModelHandler` rather than a composite:

    * ``LocalSystem.evaluate_model`` raises ``NotImplementedError`` for composite
      models, so a composite package can't be evaluated directly.
    * ORT-GenAI evaluators (``LMMSEvaluator``, ``OnnxEvaluator._inference_vision_genai``)
      dispatch on ``ONNXModelHandler`` and locate ``genai_config.json`` relative to
      the handler's ONNX file.

    This pass performs that conversion **without flattening the layout**: the nested
    directory structure is preserved (ONNX Runtime GenAI loads nested packages
    directly, and ``genai_config.json`` already references the nested paths), so no
    ONNX re-serialization or external-data rewriting is needed. The package tree is
    hardlink-copied into the pass output directory and a single
    :class:`ONNXModelHandler` is returned, pointing at the entry-point component
    (defaults to ``decoder``). Evaluators discover ``genai_config.json`` by searching
    upward from the entry ONNX file (see ``_find_genai_config`` in the evaluator).
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "entry_point_component": PassConfigParam(
                type_=str,
                default_value="decoder",
                description=(
                    "Name of the genai_config 'model' subsection (e.g. 'decoder', 'text') "
                    "whose ONNX file the returned ONNXModelHandler will point at. If the "
                    "name is not found, falls back to the first component with a 'filename' field."
                ),
            ),
        }

    @classmethod
    def is_accelerator_agnostic(cls, accelerator_spec: AcceleratorSpec) -> bool:
        # Pure file-system repackaging — no EP-specific behavior.
        return True

    def _run_for_config(
        self,
        model: CompositeModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        if not isinstance(model, CompositeModelHandler):
            raise ValueError(
                f"CompositeToOnnxPackage expects a CompositeModelHandler input, got {type(model).__name__}."
            )

        src_dir = Path(model.model_path).resolve()
        if not src_dir.is_dir():
            raise ValueError(f"CompositeModel model_path is not a directory: {src_dir}")

        src_genai_config = src_dir / "genai_config.json"
        if not src_genai_config.is_file():
            raise ValueError(
                f"CompositeToOnnxPackage requires genai_config.json at the package root: {src_genai_config} not found."
            )

        genai_config = json.loads(src_genai_config.read_text(encoding="utf-8"))
        model_section = genai_config.get("model")
        if not isinstance(model_section, dict):
            raise ValueError(f"Invalid genai_config.json at {src_genai_config}: missing 'model' section.")

        entry_filename = self._select_entry_filename(model_section, config.entry_point_component)
        if entry_filename is None:
            raise ValueError(
                "Failed to determine an entry-point component for CompositeToOnnxPackage. "
                f"Requested '{config.entry_point_component}', no component matched and no fallback available."
            )

        dst_dir = self._resolve_output_dir(output_model_path)
        # Copy the entire nested package tree (cheap hardlinks) into the pass output
        # directory, preserving the multi-component subdirectory layout as-is.
        hardlink_copy_dir(src_dir, dst_dir)
        # genai_config.json is mutable deployment metadata (provider/session
        # options are commonly adjusted after export). Give the output package an
        # independent copy so changing it cannot mutate the input pass cache.
        dst_genai_config = dst_dir / "genai_config.json"
        dst_genai_config.unlink()
        shutil.copy2(src_genai_config, dst_genai_config)

        entry_path = dst_dir / entry_filename
        if not entry_path.is_file():
            raise ValueError(
                f"Entry-point component '{entry_filename}' not found in packaged output at {entry_path}. "
                "genai_config.json references a component file that does not exist on disk."
            )

        logger.info(
            "CompositeToOnnxPackage: packaged %d components into '%s' (entry_point=%s, nested layout preserved)",
            sum(1 for v in model_section.values() if isinstance(v, dict) and v.get("filename")),
            dst_dir,
            entry_filename,
        )

        return ONNXModelHandler(
            model_path=str(dst_dir),
            onnx_file_name=entry_filename,
            model_attributes={
                "ort_genai_package": True,
                "entry_point_component": config.entry_point_component,
                **(model.model_attributes or {}),
            },
        )

    @staticmethod
    def _resolve_output_dir(output_model_path: str) -> Path:
        """Olive sometimes passes a `.onnx` file path; in that case use its stem as the directory."""
        output_path = Path(output_model_path)
        if output_path.suffix == ".onnx":
            return output_path.parent / output_path.stem
        return output_path

    @staticmethod
    def _select_entry_filename(model_section: dict, entry_point_component: str) -> str | None:
        """Pick the (nested) filename for the entry-point component, falling back if missing."""
        preferred = model_section.get(entry_point_component)
        if isinstance(preferred, dict):
            filename = preferred.get("filename")
            if isinstance(filename, str):
                return filename

        for component_cfg in model_section.values():
            if isinstance(component_cfg, dict):
                filename = component_cfg.get("filename")
                if isinstance(filename, str):
                    return filename
        return None
