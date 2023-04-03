# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List

import onnxruntime as ort
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model
from packaging import version

from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class OrtStableDiffusionOptimization(Pass):
    """Optimize stable diffusion models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description=("Must be 'unet', 'vae', or 'clip'."),
            ),
            "float16": PassConfigParam(
                type_=bool, default=True, description="Whether half-precision float will be used."
            ),
            "force_fp32_ops": PassConfigParam(
                type_=List[str], default=None, description="Operators that are forced to run in float32"
            ),
            "use_external_data_format": PassConfigParam(
                type_=bool, default=False, description="Whether use external data format to store large model (>2GB)"
            ),
        }

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        output_model_path = ONNXModel.resolve_path(output_model_path)

        config = self._config_class(**config)

        fusion_options = FusionOptions(config.model_type)
        # TODO: equivalent of fusion_options.parse(args) to add additional options from config

        use_external_data_format = False

        if config.model_type == "unet":
            # FP32 unet is too large to save into a single protobuf
            use_external_data_format = not config.float16

            # Some optimizations are not available in v1.14 or older version: packed QKV and BiasAdd
            has_all_optimizations = version.parse(ort.__version__) >= version.parse("1.15.0")
            fusion_options.enable_packed_kv = config.float16
            fusion_options.enable_packed_qkv = config.float16 and has_all_optimizations
            fusion_options.enable_bias_add = has_all_optimizations

        m = optimize_model(
            str(model.model_path),
            model_type=config.model_type,
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=True,  # TODO if "cuda" or "dml" EP
        )

        if config.float16:
            op_block_list = ["RandomNormalLike"]
            if config.force_fp32_ops:
                op_block_list += config.force_fp32_ops
            m.convert_float_to_float16(keep_io_types=False, op_block_list=op_block_list)

        m.save_model_to_file(output_model_path, use_external_data_format=use_external_data_format)

        return ONNXModel(output_model_path, model.name)
