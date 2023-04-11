# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import Any, Dict, List
from packaging import version
import onnxruntime as ort

from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class OrtTransformersOptimization(Pass):
    """Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        # TODO: add default search if supported
        return {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description=(
                    "Transformer based model type, including bert (exported by PyTorch), gpt2 (exported by PyTorch), "
                    "bert_tf (BERT exported by tf2onnx), bert_keras (BERT exported by keras2onnx), and "
                    "unet/vae/clip (stable diffusion)."
                ),
            ),
            "num_heads": PassConfigParam(type_=int, default_value=0, description="Number of attention heads."),
            "hidden_size": PassConfigParam(type_=int, default_value=0, description="Number of hidden nodes."),
            # TODO: Figure out what the expected type is
            "optimization_options": PassConfigParam(
                type_=Any, default_value=None, description="Optimization options that turn on/off some fusions."
            ),
            "opt_level": PassConfigParam(
                type_=Any,
                default_value=None,
                description=(
                    "Graph optimization level of Onnx Runtime: "
                    "0 - disable all (default), 1 - basic, 2 - extended, 99 - all."
                ),
            ),
            "use_gpu": PassConfigParam(type_=bool, default_value=False, description="Flag for GPU inference."),
            "only_onnxruntime": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether only use onnxruntime to optimize model, and no python fusion.",
            ),
            "float16": PassConfigParam(
                type_=bool, default_value=False, description="Whether half-precision float will be used."
            ),
            "input_int32": PassConfigParam(
                type_=bool, default_value=False, description="Whether int32 tensors will be used as input."
            ),
            "use_external_data_format": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether use external data format to store large model (>2GB)",
            ),
            "keep_io_types": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Keep input and output tensors in their original data type",
            ),
            "force_fp32_ops": PassConfigParam(
                type_=List[str], default=None, description="Operators that are forced to run in float32"
            ),
        }

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        from onnxruntime.transformers import optimizer as transformers_optimizer

        # start with a copy of the config
        run_config = deepcopy(config)
        del (
            run_config["float16"],
            run_config["input_int32"],
            run_config["use_external_data_format"],
            run_config["keep_io_types"],
            run_config["force_fp32_ops"],
        )

        model_type = run_config["model_type"]
        use_external_data_format = config["use_external_data_format"]

        if model_type in ("unet", "vae", "clip"):
            from onnxruntime.transformers.fusion_options import FusionOptions

            fusion_options = FusionOptions(model_type)

            if model_type == "unet":
                # FP32 unet is too large to save into a single protobuf
                use_external_data_format = not config["float16"]

                # Some optimizations are not available in v1.14 or older version: packed QKV and BiasAdd
                has_all_optimizations = version.parse(ort.__version__) >= version.parse("1.15.0")
                fusion_options.enable_packed_kv = config["float16"]
                fusion_options.enable_packed_qkv = config["float16"] and has_all_optimizations
                fusion_options.enable_bias_add = has_all_optimizations

            config["optimization_options"] = fusion_options

        optimizer = transformers_optimizer.optimize_model(input=model.model_path, **run_config)

        if config["float16"]:
            op_block_list = []
            if model_type in ("unet", "vae", "clip"):
                op_block_list += ["RandomNormalLike"]
            if config["force_fp32_ops"]:
                op_block_list += config["force_fp32_ops"]
            optimizer.convert_float_to_float16(keep_io_types=config["keep_io_types"], op_block_list=op_block_list)

        if config["input_int32"]:
            optimizer.change_graph_inputs_to_int32()

        output_model_path = ONNXModel.resolve_path(output_model_path)

        optimizer.save_model_to_file(output_model_path, use_external_data_format)

        return ONNXModel(output_model_path, model.name)
