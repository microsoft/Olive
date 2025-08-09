# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

import onnx

from olive.common.utils import hardlink_copy_dir
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class AddOliveMetadata(Pass):
    """Adds Olive-specific metadata to an ONNX model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "graph_name": PassConfigParam(
                type_=str,
                required=True,
                description="Custom graph name for the ONNX model. This field is required.",
            ),
            "custom_metadata": PassConfigParam(
                type_=dict,
                default_value={},
                description="Custom metadata key-value pairs to add to the model",
            ),
            "add_optimization_info": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Add optimization information to metadata",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _add_metadata(self, onnx_model: onnx.ModelProto, metadata: dict[str, str]) -> onnx.ModelProto:
        """Add metadata to the ONNX model."""
        # Get existing metadata
        existing_metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}

        # Update with new metadata (this will overwrite existing keys)
        existing_metadata.update(metadata)

        # Set the updated metadata
        onnx.helper.set_model_props(onnx_model, existing_metadata)

        return onnx_model

    def _set_graph_name(self, onnx_model: onnx.ModelProto, graph_name: str) -> onnx.ModelProto:
        """Set the ONNX graph name."""
        onnx_model.graph.name = graph_name
        return onnx_model

    def _generate_olive_metadata(self, model: ONNXModelHandler, config: type[BasePassConfig]) -> dict[str, str]:
        """Generate Olive-specific metadata."""
        metadata = {}

        # Add custom metadata from config
        if config.custom_metadata:
            metadata.update({str(k): str(v) for k, v in config.custom_metadata.items()})

        # Add Olive version
        try:
            import olive

            metadata["olive_version"] = getattr(olive, "__version__", "unknown")
        except Exception:
            metadata["olive_version"] = "unknown"

        # Add Hugging Face model name if available
        hf_model_name = self._get_original_hf_model_name(model)
        if hf_model_name:
            metadata["hf_model_name"] = str(hf_model_name)

        # Add optimization information
        if config.add_optimization_info and hasattr(model, "model_attributes") and model.model_attributes:
            model_attrs = model.model_attributes

            # Add optimization passes applied
            if "optimization_passes" in model_attrs:
                passes = model_attrs["optimization_passes"]
                if isinstance(passes, list):
                    metadata["optimization_passes"] = ", ".join(str(p) for p in passes)
                else:
                    metadata["optimization_passes"] = str(passes)

            # Add HuggingFace task if available
            if "hf_task" in model_attrs:
                metadata["hf_task"] = str(model_attrs["hf_task"])

            # Add any quantization information
            if "quantization_config" in model_attrs:
                metadata["quantization_config"] = str(model_attrs["quantization_config"])

        return metadata

    def _get_original_hf_model_name(self, model: ONNXModelHandler) -> str:
        """Get the original HF model name from the model's config if the original model was a HuggingFace model."""
        try:
            model_json = model.to_json()
            config = model_json.get("config", {})
            model_attrs = config.get("model_attributes", {})

            # Check if the original model was a HuggingFace model
            model_type = model_attrs.get("type", "").lower()
            if model_type in ["hfmodel", "hf_model"]:
                # Try to get model path from _name_or_path in model_attributes
                hf_model_name = model_attrs.get("_name_or_path")
                if hf_model_name and isinstance(hf_model_name, str):
                    return hf_model_name

        except Exception as e:
            logger.warning("Could not extract original HF model name from model config: %s", e)

        return None

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise ValueError("Model must be an instance of ONNXModelHandler")

        from pathlib import Path

        input_onnx_file = Path(model.model_path)
        output_path = Path(output_model_path)

        # Determine input model directory
        if input_onnx_file.is_dir():
            # This shouldn't happen since model.model_path should resolve to the ONNX file
            raise ValueError(f"Model path resolved to directory, expected ONNX file: {input_onnx_file}")

        input_model_dir = input_onnx_file.parent

        # Load ONNX model without external data to preserve file structure
        onnx_model = onnx.load_model(str(input_onnx_file), load_external_data=False)

        # Generate metadata
        metadata = self._generate_olive_metadata(model, config)

        # Calculate model hash
        try:
            from olive.model.config.model_config import ModelConfig

            model_config = ModelConfig.parse_obj(model.to_json())
            model_hash = model_config.get_model_identifier()
            metadata["model_hash"] = model_hash
        except Exception as e:
            logger.warning("Could not calculate model hash: %s", e)
            metadata["model_hash"] = "unknown"

        # Add metadata and set graph name
        if metadata:
            onnx_model = self._add_metadata(onnx_model, metadata)
        onnx_model = self._set_graph_name(onnx_model, config.graph_name)

        # Determine output directory
        if output_path.suffix == ".onnx":
            output_dir = output_path.parent / output_path.stem
        else:
            output_dir = output_path

        # Copy entire input directory to preserve external files using hardlinks
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        hardlink_copy_dir(input_model_dir, output_dir)

        # Save updated ONNX file
        output_onnx_file = output_dir / input_onnx_file.name
        if output_onnx_file.exists():
            # Remove the hard-linked file to break the connection to the original
            output_onnx_file.unlink()

        # Save updated ONNX file (now it's a new independent file)
        onnx.save_model(onnx_model, str(output_onnx_file))

        return ONNXModelHandler(model_path=output_dir, onnx_file_name=input_onnx_file.name)
