#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from pathlib import Path
from typing import Optional

import onnx

from olive.constants import Precision
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file, resave_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def dtype_from_precision(p: Precision) -> Optional[str]:
    """Convert Precision enum to VAI metadata data type string."""
    mapping = {
        Precision.INT4: "Int4",
        # is this UInt4 or QUInt4?
        Precision.UINT4: "UInt4",
        Precision.INT8: "QInt8",
        Precision.UINT8: "QUInt8",
        Precision.INT16: "QInt16",
        Precision.UINT16: "QUInt16",
    }
    return mapping.get(p)


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
                type_=Precision,
                required=False,
                description="Activation dytpe",
            ),
            "weight_type": PassConfigParam(
                type_=Precision,
                required=False,
                description="Weight dtype",
            ),
            "quant_type": PassConfigParam(type_=str, required=False, description="Quant dtype", default_value="NA"),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.activation_type and not dtype_from_precision(config.activation_type):
            logger.warning("Unsupported activation type: %s", config.activation_type)
            return False

        if config.weight_type and not dtype_from_precision(config.weight_type):
            logger.warning("Unsupported weight type: %s", config.weight_type)
            return False

        return True

    def _add_meta_data(self, onnx_model: onnx.ModelProto, metadata: dict) -> onnx.ModelProto:
        """Add the meta data to the model proto."""
        new_metadata_props = {entry.key: entry.value for entry in onnx_model.metadata_props}
        # update the metadata properties, this will overwrite the existing ones
        new_metadata_props.update(metadata)
        onnx.helper.set_model_props(onnx_model, new_metadata_props)
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
            metadata["activation_dtype"] = dtype_from_precision(config.activation_type)
        if config.weight_type:
            metadata["weight_dtype"] = dtype_from_precision(config.weight_type)
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
        # save the model with metadata, will unlink to avoid modifying the hardlinked original
        model_proto_to_file(onnx_model, output_model_path)

        return ONNXModelHandler(
            model_path=output_model_path.parent if has_external_data else output_model_path,
            onnx_file_name=output_model_path.name if has_external_data else None,
        )
