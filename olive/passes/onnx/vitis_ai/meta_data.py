#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path
from typing import Optional

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VitisAIAddMetaData(Pass):
    """Adds metadata to an ONNX model based on specified model attributes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "config_meta_data_keys": PassConfigParam(
                type_=list,
                required=False,
                description="List of model attribute keys to extract as metadata",
                default_value=["architectures", "model_type"],
            ),
            "activation_type": PassConfigParam(
                type_=str,
                required=False,
                description="Activation dytpe",
            ),
            "weight_type": PassConfigParam(
                type_=str,
                required=False,
                description="weight dtype",
            ),
            "quant_type": PassConfigParam(type_=str, required=False, description="Quant dtype", default_value="NA"),
        }

    def _get_component_group(self, model: CompositeModelHandler, pattern: str) -> list[str]:
        """Identify components by naming pattern or position."""
        # Example implementation - adjust based on your actual naming conventions
        return [name for name in model.model_component_names if pattern in name.lower()]

    def _add_meta_data(self, onnx_model: onnx.ModelProto, metadata: dict) -> onnx.ModelProto:
        """Add the meta data to the model proto."""
        existing_metadata = {entry.key: idx for idx, entry in enumerate(onnx_model.metadata_props)}

        for key in metadata:
            if key in existing_metadata:
                del onnx_model.metadata_props[existing_metadata[key]]

        # Add validated metadata
        for key, value in metadata.items():
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = str(value)
        return onnx_model

    def _run_for_config(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise ValueError("Model must be an instance of ONNXModelHandler")

        if not hasattr(model, "model_attributes") or not model.model_attributes:
            raise ValueError("Model attributes are missing")

        model_details = model.model_attributes
        config_meta_data_keys = config.config_meta_data_keys

        def get_attribute(key: str) -> Optional[str]:
            value = model_details.get(key)
            if isinstance(value, list):
                return ", ".join(map(str, value))
            return str(value) if value is not None else None

        # Prepare metadata
        metadata = {key: get_attribute(key) for key in config_meta_data_keys if get_attribute(key) is not None}

        if config.activation_type:
            metadata["activation_dtype"] = config.activation_type
        if config.weight_type:
            metadata["weight_dtype"] = config.weight_type
        if config.quant_type:
            metadata["quant_type"] = config.quant_type

        if not metadata:
            logger.warning("No metadata to add to the model.")
            return model

        output_model_path = Path(resolve_onnx_path(output_model_path, Path(model.model_path).name))

        # resave the original model to the new path
        has_external_data = resave_model(model.model_path, output_model_path)
        # load the model without external data
        onnx_model = self._add_meta_data(onnx.load_model(output_model_path, load_external_data=False), metadata)
        # save the model with metadata
        onnx.save_model(onnx_model, output_model_path)

        return ONNXModelHandler(
            model_path=output_model_path.parent if has_external_data else output_model_path,
            onnx_file_name=output_model_path.name if has_external_data else None,
        )
