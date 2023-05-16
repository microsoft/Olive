# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import Any, Dict, List, Union

from packaging import version

from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class OrtTransformersOptimization(Pass):
    """Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        # TODO: add default search if supported
        from onnxruntime.transformers.fusion_options import FusionOptions

        config = {
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
                type_=Union[Dict[str, Any], FusionOptions],
                default_value=None,
                description="Optimization options that turn on/off some fusions.",
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
            "keep_io_types": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Keep input and output tensors in their original data type",
            ),
            "force_fp32_ops": PassConfigParam(
                type_=List[str], default_value=None, description="Operators that are forced to run in float32"
            ),
            "target_provider": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Target execution provider. This parameter will be removed when "
                    "accelerators/targets are visible to passes."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    @staticmethod
    def _set_fusion_options(run_config: Dict[str, Any]):
        from onnxruntime.transformers.fusion_options import FusionOptions

        fusion_options = FusionOptions(run_config["model_type"])
        fusion_options.__dict__.update(run_config["optimization_options"])
        run_config["optimization_options"] = fusion_options

    @staticmethod
    def sd_model_types():
        """Returns model types in the stable diffusion pipeline recognized by the ORT transformer optimizer"""
        return ("unet", "vae", "clip")

    @staticmethod
    def _set_sd_fusion_options(run_config: Dict[str, Any], pass_config: Dict[str, Any]):
        """Configures fusion options for stable diffusion models"""
        import onnxruntime as ort

        ort_version = version.parse(ort.__version__)
        is_ort_1_13_or_older = ort_version < version.parse("1.14.0")
        # is_ort_1_14 = ort_version >= version.parse("1.14.0") and ort_version < version.parse("1.15.0")

        # default to no specific fusion options in earlier releases of ORT
        if is_ort_1_13_or_older:
            return

        is_ort_1_15_0_or_newer = ort_version >= version.parse("1.15.0")
        is_ort_1_15_1_or_newer = ort_version >= version.parse("1.15.1")

        input_model_type = run_config["model_type"]
        if not is_ort_1_15_0_or_newer and input_model_type != "unet":
            # 'vae' and 'clip' are only recognized in ORT v1.15+. earlier versions of ORT stable diffusion
            # optimization simply use "unet" for these model types.
            run_config["model_type"] = "unet"

        from onnxruntime.transformers.fusion_options import FusionOptions

        fusion_options = FusionOptions(run_config["model_type"])

        # TODO: remove dml_future when ORT 1.15.1 with future version of DML is released
        # This "provider" value is simply a way to test in-development changes without having
        # to bump the ORT version.
        dml_future = pass_config["target_provider"] == "directml_future"

        if pass_config["target_provider"] == "directml" or dml_future:
            # Some of these fusions are disabled because they provide no performance advantage,
            # and it's preferable to limit ops outside the ONNX domain.
            fusion_options.enable_gelu = is_ort_1_15_0_or_newer
            fusion_options.enable_layer_norm = is_ort_1_15_0_or_newer
            fusion_options.enable_attention = is_ort_1_15_1_or_newer or dml_future
            fusion_options.use_multi_head_attention = is_ort_1_15_1_or_newer or dml_future
            fusion_options.enable_skip_layer_norm = False
            fusion_options.enable_embed_layer_norm = is_ort_1_15_0_or_newer
            fusion_options.enable_bias_skip_layer_norm = False
            fusion_options.enable_bias_gelu = is_ort_1_15_0_or_newer
            fusion_options.enable_gelu_approximation = False
            fusion_options.enable_qordered_matmul = False
            fusion_options.enable_shape_inference = is_ort_1_15_0_or_newer
            fusion_options.enable_gemm_fast_gelu = False
            fusion_options.enable_nhwc_conv = False
            fusion_options.enable_group_norm = is_ort_1_15_1_or_newer or dml_future
            fusion_options.enable_bias_splitgelu = False
            fusion_options.enable_packed_qkv = pass_config["float16"] and is_ort_1_15_0_or_newer
            fusion_options.enable_packed_kv = pass_config["float16"] and is_ort_1_15_0_or_newer
            fusion_options.enable_bias_add = False
        else:
            if input_model_type == "unet":
                fusion_options.enable_packed_kv = pass_config["float16"]
                fusion_options.enable_packed_qkv = pass_config["float16"] and is_ort_1_15_0_or_newer
                fusion_options.enable_bias_add = is_ort_1_15_0_or_newer

        run_config["optimization_options"] = fusion_options

    @staticmethod
    def _get_op_block_list(config: Dict[str, Any]):
        import onnxruntime as ort

        op_block_list = []

        if config["float16"]:
            if config["model_type"] in OrtTransformersOptimization.sd_model_types():
                if version.parse(ort.__version__) < version.parse("1.15.0"):
                    op_block_list += ["RandomNormalLike", "Resize", "GroupNorm"]
                else:
                    op_block_list += ["RandomNormalLike"]
            if config["force_fp32_ops"]:
                op_block_list += config["force_fp32_ops"]

        return op_block_list if len(op_block_list) > 0 else None

    @staticmethod
    def _run_without_optimization(model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        """Fallback/pass-through path that skips onnxruntime.transformers.optimizer and simply copies the model
        or converts it to FP16 without performing transformer-specific optimizations. This fallback exists so
        that a pass configuration can be used with multiple ORT versions, some of which may not support optimization.
        """

        if not config["float16"]:
            return model

        import onnx
        from onnxconverter_common import float16

        # stable diffusion unet is too large for the converter to handle with shape inference
        disable_shape_infer = config["model_type"] == "unet"

        op_block_list = OrtTransformersOptimization._get_op_block_list(config)

        model_fp32 = onnx.load(str(model.model_path))
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            keep_io_types=config["keep_io_types"],
            op_block_list=op_block_list,
            disable_shape_infer=disable_shape_infer,
        )
        # save the model to the output path and return the model
        return model_proto_to_olive_model(model_fp16, output_model_path, config, model.name, model.model_file_format)

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer as transformers_optimizer

        # start with a copy of the config
        run_config = deepcopy(config)
        del (
            run_config["float16"],
            run_config["input_int32"],
            run_config["keep_io_types"],
            run_config["force_fp32_ops"],
            run_config["target_provider"],
        )
        for key in get_external_data_config():
            del run_config[key]

        output_model_path = ONNXModel.resolve_path(output_model_path)

        if config["model_type"] in OrtTransformersOptimization.sd_model_types():
            # stable diffusion optimization only applies to the CUDA EP in ORT 1.14 and earlier.
            if config["target_provider"] != "cuda" and version.parse(ort.__version__) < version.parse("1.15.0"):
                return self._run_without_optimization(model, config, output_model_path)

            if config["optimization_options"] is None:
                self._set_sd_fusion_options(run_config, config)
        elif config["optimization_options"]:
            self._set_fusion_options(run_config)

        optimizer = transformers_optimizer.optimize_model(input=model.model_path, **run_config)

        if config["float16"]:
            op_block_list = self._get_op_block_list(config)
            optimizer.convert_float_to_float16(keep_io_types=config["keep_io_types"], op_block_list=op_block_list)

        if config["input_int32"]:
            optimizer.change_graph_inputs_to_int32()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(
            optimizer.model, output_model_path, config, model.name, model.model_file_format
        )
