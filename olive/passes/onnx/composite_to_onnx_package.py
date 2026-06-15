# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convert a multi-component CompositeModel ORT-GenAI package into a flat ONNX package."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

from olive.common.utils import hardlink_copy_file
from olive.model import ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec

logger = logging.getLogger(__name__)


class CompositeToOnnxPackage(Pass):
    """Flatten a CompositeModel ORT-GenAI package into a single ONNXModel handler.

    MobiusBuilder and similar passes emit multi-component ORT-GenAI packages as a
    :class:`CompositeModelHandler` whose components live in subdirectories::

        output_dir/
          genai_config.json
          tokenizer.json
          decoder/model.onnx
          vision_encoder/model.onnx
          audio_encoder/model.onnx
          embedding/model.onnx

    Olive's evaluators (e.g. ``OnnxEvaluator._inference_vision_genai`` and the
    ``LMMSEvaluator``) detect ORT-GenAI packages by looking for ``genai_config.json``
    next to an ONNX file referenced by an :class:`ONNXModelHandler`. The nested
    subdirectory layout above defeats that detection because the entry-point ONNX
    file's parent (e.g. ``output_dir/decoder/``) does not contain
    ``genai_config.json``.

    This pass produces an equivalent flat layout::

        output_dir/
          genai_config.json
          tokenizer.json
          decoder.onnx
          vision_encoder.onnx
          audio_encoder.onnx
          embedding.onnx

    by hardlinking each component (and its ``.onnx.data`` sidecar, if present) to
    the package root and rewriting ``genai_config.json`` to reference the flat
    filenames. The returned :class:`ONNXModelHandler` points at the entry-point
    component (defaults to ``decoder``), so downstream evaluators can auto-detect
    the package via ``Path(model.model_path).parent / "genai_config.json"``.
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
        # Pure file-system / config rewrite — no EP-specific behavior.
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

        dst_dir = self._resolve_output_dir(output_model_path)
        dst_dir.mkdir(parents=True, exist_ok=True)

        genai_config = json.loads(src_genai_config.read_text(encoding="utf-8"))
        model_section = genai_config.get("model")
        if not isinstance(model_section, dict):
            raise ValueError(f"Invalid genai_config.json at {src_genai_config}: missing 'model' section.")

        rewrite_map = self._build_rewrite_map(model_section)
        if not rewrite_map:
            raise ValueError(
                f"No component subsections with 'filename' found in genai_config.json at {src_genai_config}."
            )

        # Copy each component ONNX into the flat layout, rewriting external-data
        # references so each initializer points at the renamed sidecar.
        #
        # We can't just hardlink the .onnx file and its .data sidecar to the new
        # names, because each ONNX file embeds the external-data filename
        # ("location" entry in the proto). After renaming, those embedded
        # pointers still reference the old name (e.g. "model.onnx.data") and
        # ONNX Runtime fails at load with "External data path does not exist".
        # ``onnx.save_model(..., save_as_external_data=True, location=...)``
        # serializes a new ONNX file whose embedded location matches the new
        # filename, and writes the corresponding .data file alongside.
        for old_rel, new_name in rewrite_map.items():
            src_file = self._resolve_component_source(src_dir, old_rel)
            if src_file is None:
                raise ValueError(f"Component file referenced by genai_config not found: {src_dir / old_rel}")

            src_data = self._resolve_component_data(src_file)
            dst_file = dst_dir / new_name
            dst_data_name = f"{new_name}.data"
            dst_data_file = dst_dir / dst_data_name

            if src_data is not None:
                # Load model and resolve external initializer tensors so we can
                # re-serialize them under the new filename. ``load_external_data
                # =True`` (the default) materializes initializer bytes into the
                # in-memory proto via the source directory layout, after which
                # we can write them back out with a new ``location``.
                onnx_model = onnx.load(str(src_file), load_external_data=True)
                # Remove any pre-existing destination files to avoid onnx
                # appending to a stale .data sidecar on rerun.
                if dst_file.exists():
                    dst_file.unlink()
                if dst_data_file.exists():
                    dst_data_file.unlink()
                onnx.save_model(
                    onnx_model,
                    str(dst_file),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=dst_data_name,
                )
            else:
                # No external data sidecar — model is self-contained, plain copy.
                hardlink_copy_file(src_file, dst_file)

        # Copy every top-level shared sidecar (tokenizer, processor configs, chat template, etc.).
        # genai_config.json is rewritten below, so skip it here.
        for src_file in src_dir.iterdir():
            if not src_file.is_file() or src_file.name == "genai_config.json":
                continue
            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                hardlink_copy_file(src_file, dst_file)

        # Update filename references and write the rewritten config.
        for component_cfg in model_section.values():
            if isinstance(component_cfg, dict):
                old_name = component_cfg.get("filename")
                if isinstance(old_name, str) and old_name in rewrite_map:
                    component_cfg["filename"] = rewrite_map[old_name]

        (dst_dir / "genai_config.json").write_text(
            json.dumps(genai_config, indent=2),
            encoding="utf-8",
        )

        entry_filename = self._select_entry_filename(model_section, config.entry_point_component)
        if entry_filename is None:
            raise ValueError(
                "Failed to determine an entry-point component for CompositeToOnnxPackage. "
                f"Requested '{config.entry_point_component}', no component matched and no fallback available."
            )

        logger.info(
            "CompositeToOnnxPackage: flattened %d components into '%s' (entry_point=%s)",
            len(rewrite_map),
            dst_dir,
            entry_filename,
        )

        return ONNXModelHandler(
            model_path=str(dst_dir),
            onnx_file_name=entry_filename,
            model_attributes={
                "ort_genai_package": True,
                "entry_point_component": config.entry_point_component,
                "flattened_from_composite": True,
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
    def _resolve_component_source(src_dir: Path, old_rel: str) -> Path | None:
        """Resolve the on-disk source file for a component referenced by genai_config.

        Some upstream Olive passes (notably ``OnnxKQuantQuantization`` when given a
        component already named ``decoder.onnx``) save the quantized model with
        the ``.onnx`` extension stripped — producing ``decoder``/``encoder`` files
        next to ``decoder.data``/``encoder.data`` while ``genai_config.json`` still
        references the original ``decoder.onnx``/``encoder.onnx``. Accept the
        extensionless variant so we can still flatten such packages without
        requiring an upstream fix.
        """
        candidate = src_dir / old_rel
        if candidate.is_file():
            return candidate
        stripped = src_dir / Path(old_rel).stem
        if stripped.is_file():
            return stripped
        return None

    @staticmethod
    def _resolve_component_data(src_file: Path) -> Path | None:
        """Resolve the external-data sidecar for a component source file.

        Tries ``<src>.data`` first (matches ONNX's default ``<filename>.data``
        sidecar). For extensionless source files emitted by buggy upstream
        passes, also accepts ``<src_stem>.data`` (e.g. ``decoder`` + ``decoder.data``).
        """
        primary = src_file.with_name(src_file.name + ".data")
        if primary.is_file():
            return primary
        if src_file.suffix == "":
            alt = src_file.with_name(src_file.stem + ".data")
            if alt.is_file():
                return alt
        return None

    @staticmethod
    def _build_rewrite_map(model_section: dict) -> dict[str, str]:
        """Map each old relative filename to a unique flat root-level filename.

        Uses the immediate parent directory name when the component lives in a
        subdirectory (``decoder/model.onnx`` -> ``decoder.onnx``). Falls back to
        the genai_config key when the file is already flat or the parent name
        collides. Guarantees uniqueness by appending a counter if needed.
        """
        used_names: set[str] = set()
        rewrite_map: dict[str, str] = {}

        for component_key, component_cfg in model_section.items():
            if not isinstance(component_cfg, dict):
                continue
            old_path = component_cfg.get("filename")
            if not isinstance(old_path, str) or not old_path:
                continue
            if old_path in rewrite_map:
                continue

            old_path_obj = Path(old_path)
            parent_name = old_path_obj.parent.name
            candidate_base = parent_name or component_key
            candidate = f"{candidate_base}.onnx"

            if candidate in used_names:
                suffix = 1
                while f"{candidate_base}_{suffix}.onnx" in used_names:
                    suffix += 1
                candidate = f"{candidate_base}_{suffix}.onnx"

            used_names.add(candidate)
            rewrite_map[old_path] = candidate

        return rewrite_map

    @staticmethod
    def _select_entry_filename(model_section: dict, entry_point_component: str) -> str | None:
        """Pick the flat filename for the entry-point component, falling back if missing."""
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
