# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from typing import Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.handler.multi_target import MultiTargetModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class EPContextBinaryPackager(Pass):
    """Generate a manifest.json metadata file for multi-target EP context binaries.

    This pass takes a MultiTargetModelHandler (produced by EPContextBinaryGenerator with
    a list of provider_options) and generates a manifest.json file describing each target's
    context binary with metadata required by ONNX Runtime.

    The manifest includes:
    - ep: execution provider name
    - device_type: CPU, NPU, or GPU
    - architecture: hardware architecture (e.g., SoC model)
    - precision: model precision (from model_attributes)
    - sdk_version: optional SDK version
    - compile_options: optional compilation options
    """

    _accepts_composite_model = True
    _accepts_multi_target_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description="Model name for the manifest. If not set, derived from the output directory name.",
            ),
            "sdk_version": PassConfigParam(
                type_=str,
                default_value=None,
                description="SDK version string (e.g., 'qnn_sdk_2.28').",
            ),
            "compile_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Additional compile options to include in the manifest (e.g., dynamic shape, batch size).",
            ),
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: MultiTargetModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> MultiTargetModelHandler:
        assert isinstance(model, MultiTargetModelHandler), (
            "EPContextBinaryPackager requires a MultiTargetModelHandler as input. "
            "Use EPContextBinaryGenerator with a list of provider_options to produce one."
        )

        output_dir = Path(output_model_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive model name from config or output directory
        model_name = config.model_name or output_dir.name

        manifest = {"name": model_name, "components": []}

        for target_name, target_model in model.get_target_models():
            target_attrs = target_model.model_attributes or {}

            # Copy target model files to output directory
            self._copy_target_model(target_name, target_model, output_dir)

            # Determine the model path relative to output directory
            model_path = self._get_relative_model_path(target_name, target_model)

            entry = {
                "variant_name": target_name,
                "file": model_path,
                "constraints": {
                    "ep": self.accelerator_spec.execution_provider,
                    "device": target_attrs.get("target_device", str(self.accelerator_spec.accelerator_type).upper()),
                    "architecture": target_attrs.get("architecture", target_name),
                },
            }

            # Add precision from model_attributes if available
            precision = target_attrs.get("precision")
            if precision:
                entry["constraints"]["precision"] = precision

            # Add sdk_version from model_attributes or config
            sdk_version = target_attrs.get("sdk_version") or config.sdk_version
            if sdk_version:
                entry["constraints"]["sdk_version"] = sdk_version
            if config.compile_options:
                entry["constraints"]["compile_options"] = config.compile_options

            manifest["components"].append(entry)

        # Write manifest.json
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Generated manifest at %s", manifest_path)

        # Update model_attributes to include manifest path
        # Remove additional_files since each target subfolder already contains its own tokenizer/config files
        new_model_attributes = model.model_attributes or {}
        new_model_attributes = {**new_model_attributes, "manifest_path": str(manifest_path)}
        new_model_attributes.pop("additional_files", None)

        # Return the same MultiTargetModelHandler with updated attributes and path
        return MultiTargetModelHandler(
            [target_model for _, target_model in model.get_target_models()],
            [target_name for target_name, _ in model.get_target_models()],
            model_path=output_dir,
            model_attributes=new_model_attributes,
        )

    @staticmethod
    def _copy_target_model(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
        output_dir: Path,
    ) -> None:
        """Copy target model files to the output directory under target_name/."""
        dest_dir = output_dir / target_name
        if dest_dir.exists():
            return

        if isinstance(target_model, CompositeModelHandler):
            src_dir = Path(target_model.model_path)
        else:
            src_dir = Path(target_model.model_path).parent

        if src_dir.is_dir():
            shutil.copytree(str(src_dir), str(dest_dir))
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target_model.model_path), str(dest_dir))

    @staticmethod
    def _get_relative_model_path(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
    ) -> str:
        """Get the model path relative to the target name for the manifest."""
        if isinstance(target_model, ONNXModelHandler):
            return f"{target_name}/{Path(target_model.model_path).name}"
        # For CompositeModelHandler or other types, use the directory
        return f"{target_name}/"
