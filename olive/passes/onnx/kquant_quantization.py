# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""K-quant weight-only quantization pass using OnnxRuntime's MatMulNBitsQuantizer."""

import logging
from pathlib import Path

from olive.constants import AccuracyLevel
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

# Minimum ORT version required for KQuantWeightOnlyQuantConfig
_MIN_ORT_VERSION = "1.20.0"


class OnnxKQuantQuantization(Pass):
    """Quantize ONNX models with k-quant algorithm via MatMulNBitsQuantizer.

    Uses onnxruntime's KQuantWeightOnlyQuantConfig to produce INT4 MatMulNBits
    nodes. K-quant assigns different quantization bit-widths to different weight
    matrices based on their sensitivity, achieving better accuracy than uniform
    RTN quantization at the same average bit-width.

    Use ``customized_weight_config`` to assign per-node quantization settings
    (e.g., sensitive layers at INT8 while others use INT4).
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "block_size": PassConfigParam(
                type_=int,
                default_value=32,
                description="Block size for quantization. Default value is 32.",
            ),
            "is_symmetric": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to use symmetric quantization. Default value is True.",
            ),
            "accuracy_level": PassConfigParam(
                type_=AccuracyLevel,
                default_value=AccuracyLevel.unset,
                description=(
                    "Accuracy level of the 4-bit quantized MatMul computation. Refer to the MatMulNBits"
                    " contrib op's 'accuracy_level' attribute for details"
                    " (https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md"
                    "#commicrosoftmatmulnbits)."
                ),
            ),
            "customized_weight_config": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Per-node quantization overrides. A dict mapping node names to their config, "
                    'e.g. {"node_name": {"bits": 8}} to quantize sensitive nodes at INT8 '
                    "while the rest use INT4. Passed directly to KQuantWeightOnlyQuantConfig."
                ),
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to exclude from quantization.",
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        import onnx

        try:
            from onnxruntime import __version__ as ort_version
            from onnxruntime.quantization.matmul_nbits_quantizer import (
                KQuantWeightOnlyQuantConfig,
                MatMulNBitsQuantizer,
            )
        except ImportError as e:
            raise ImportError(
                f"onnxruntime >= {_MIN_ORT_VERSION} is required for OnnxKQuantQuantization. "
                "Please install it with: pip install onnxruntime"
            ) from e

        from packaging import version

        if version.parse(ort_version) < version.parse(_MIN_ORT_VERSION):
            raise ValueError(
                f"OnnxKQuantQuantization requires onnxruntime >= {_MIN_ORT_VERSION}, "
                f"but found {ort_version}. Please upgrade: pip install --upgrade onnxruntime"
            )

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        logger.info("Loading ONNX model from %s for k-quant quantization", model.model_path)
        onnx_model = onnx.load(str(model.model_path), load_external_data=True)

        if config.customized_weight_config:
            logger.info(
                "Using customized weight config for %d nodes",
                len(config.customized_weight_config),
            )
            algo_config = KQuantWeightOnlyQuantConfig(
                customized_weight_config=config.customized_weight_config
            )
        else:
            algo_config = KQuantWeightOnlyQuantConfig()

        quantizer = MatMulNBitsQuantizer(
            model=onnx_model,
            block_size=config.block_size,
            is_symmetric=config.is_symmetric,
            accuracy_level=config.accuracy_level if config.accuracy_level > 0 else None,
            nodes_to_exclude=config.nodes_to_exclude or [],
            algo_config=algo_config,
        )
        quantizer.process()

        return model_proto_to_olive_model(quantizer.model.model, output_model_path, config)
