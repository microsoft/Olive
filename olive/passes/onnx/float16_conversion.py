# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Type

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class OnnxFloatToFloat16(Pass):
    """Converts a model to float16.

    It uses the float16 converter from onnxruntime to convert the model to float16.
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
            "use_symbolic_shape_infer": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Use symbolic shape inference instead of onnx shape inference. Defaults to True.",
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
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime.transformers.onnx_model import OnnxModel

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # using the float16 converter from onnxruntime since it is regularly updated
        # and can handle large models (>2GB) as well as ort contrib ops
        ort_onnx_model = OnnxModel(model.load_model())
        config_dict = config.dict()
        ort_onnx_model.convert_float_to_float16(
            **{
                key: config_dict[key]
                for key in [
                    "min_positive_val",
                    "max_finite_val",
                    "keep_io_types",
                    "use_symbolic_shape_infer",
                    "op_block_list",
                    "node_block_list",
                ]
            }
        )

        # save the model to the output path and return the model
        return model_proto_to_olive_model(ort_onnx_model.model, output_model_path, config)
