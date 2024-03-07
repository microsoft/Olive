# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class OnnxFloatToFloat16(Pass):
    """Converts a model to float16.

    It is based on onnxconverter-common.convert_float_to_float16.
    See https://onnxruntime.ai/docs/performance/model-optimizations/float16.html#float16-conversion
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "min_positive_val": PassConfigParam(
                type_=float, default_value=1e-7, description="Constant values will be clipped against this value"
            ),
            "max_finite_val": PassConfigParam(
                type_=float, default_value=1e4, description="Constant values will be clipped against this value"
            ),
            "keep_io_types": PassConfigParam(
                type_=bool, default_value=False, description="Whether model inputs/outputs should be left as float32"
            ),
            "disable_shape_infer": PassConfigParam(
                type_=bool, default_value=False, description="Skips running onnx shape/type inference."
            ),
            "op_block_list": PassConfigParam(
                type_=List[str], default_value=None, description="List of op types to leave as float32"
            ),
            "node_block_list": PassConfigParam(
                type_=List[str], default_value=None, description="List of node names to leave as float32"
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxconverter_common import float16

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        config = self._config_class(**config)

        model_fp32 = onnx.load(str(model.model_path))
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            min_positive_val=config.min_positive_val,
            max_finite_val=config.max_finite_val,
            keep_io_types=config.keep_io_types,
            disable_shape_infer=config.disable_shape_infer,
            op_block_list=config.op_block_list,
            node_block_list=config.node_block_list,
        )

        # save the model to the output path and return the model
        return model_proto_to_olive_model(model_fp16, output_model_path, config.dict())
