# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file, resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class AddOliveMetadata(Pass):
    """Adds Olive-specific metadata to an ONNX model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
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
            "add_olive_version": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Add Olive version to metadata",
            ),
            "add_optimization_info": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Add optimization information to metadata",
            ),
        }

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

    def _generate_olive_metadata(self, model: ONNXModelHandler, config: BasePassConfig) -> dict[str, str]:
        """Generate Olive-specific metadata."""
        metadata = {}

        # Add custom metadata from config
        if config.custom_metadata:
            metadata.update({str(k): str(v) for k, v in config.custom_metadata.items()})

        # Add Olive version
        if config.add_olive_version:
            try:
                import olive

                metadata["olive_version"] = getattr(olive, "__version__", "unknown")
            except Exception:
                metadata["olive_version"] = "unknown"

        # Add optimization information
        if config.add_optimization_info and hasattr(model, "model_attributes") and model.model_attributes:
            model_attrs = model.model_attributes

            # Add original model information
            if "original_model_path" in model_attrs:
                metadata["original_model_path"] = str(model_attrs["original_model_path"])

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

    def _run_for_config(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise ValueError("Model must be an instance of ONNXModelHandler")

        # Validate that graph_name is provided
        if not config.graph_name:
            raise ValueError("graph_name is required and must be provided in the AddOliveMetadata pass configuration")

        # Generate metadata
        metadata = self._generate_olive_metadata(model, config)

        # Get graph name from config
        graph_name = config.graph_name

        # If no metadata to add, we still need to set the graph name since it's mandatory
        if not metadata:
            logger.info("No metadata to add, but will still set the mandatory graph name.")

        output_model_path = Path(resolve_onnx_path(output_model_path, Path(model.model_path).name))

        # Resave the original model to the new path
        has_external_data = resave_model(model.model_path, output_model_path)

        # Load the model without external data to modify metadata
        onnx_model = onnx.load_model(output_model_path, load_external_data=False)

        # Add metadata
        if metadata:
            logger.info("Adding metadata to ONNX model: %s", metadata)
            onnx_model = self._add_metadata(onnx_model, metadata)

        # Set graph name (always required)
        logger.info("Setting ONNX graph name to: %s", graph_name)
        onnx_model = self._set_graph_name(onnx_model, graph_name)

        # Save the model with updated metadata
        model_proto_to_file(onnx_model, output_model_path)

        return ONNXModelHandler(
            model_path=output_model_path.parent if has_external_data else output_model_path,
            onnx_file_name=output_model_path.name if has_external_data else None,
        )
