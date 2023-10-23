# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, List

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class OnnxBnb4Quantization(Pass):
    """Quantize MatMul nodes in ONNX model using 4bit FP4/NF4 quantization."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "quant_type": PassConfigParam(
                type_=str,
                description="The quantization type. Only 'fp4' and 'nf4' are supported.",
            ),
            "quantized_modules": PassConfigParam(
                type_=List[str],
                description=(
                    "The list of modules to quantize. Will match the end of MatMul nodes with this list to determine"
                    " which nodes to quantize."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        from onnxruntime.quantization import MatMulBnb4Quantizer

        output_model_path = ONNXModel.resolve_path(output_model_path)

        quant_type = config["quant_type"]
        quantized_modules = config["quantized_modules"]
        if model.model_attributes:
            quantized_modules = quantized_modules or model.model_attributes.get("quantized_modules")

            # extract quant_type from model_attributes if not specified in config
            if not quant_type:
                quant_config = model.model_attributes.get("quantization_config")

                if quant_config.get("load_in_8bit"):
                    raise ValueError("load_in_8bit is not supported. Only 4-bit quantization is supported.")

                quant_type = quant_config.get("bnb_4bit_quant_type")
                if not quant_type:
                    raise ValueError("quant_type is required.")

                if quant_config.get("bnb_4bit_use_double_quant"):
                    logger.info(
                        "bnb_4bit_use_double_quant is set to True but double quantization is not supported. Ignoring."
                    )
        assert quant_type in ["fp4", "nf4"], f"quant_type must be one of 'fp4' or 'nf4'. Got {quant_type}."
        quant_type_enum = getattr(MatMulBnb4Quantizer, quant_type.upper())

        onnx_model = model.load_model()
        quantizer = MatMulBnb4Quantizer(
            onnx_model, quant_type=quant_type_enum, block_size=64, nodes_filter=quantized_modules
        )
        quantizer.process()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)
