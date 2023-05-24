# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from copy import deepcopy
from typing import Any, Dict, List, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class OrtTransformersOptimization(Pass):
    """Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        }
        config.update(get_external_data_config())
        return config

    @staticmethod
    def _set_fusion_options(run_config: Dict[str, Any]):
        from onnxruntime.transformers.fusion_options import FusionOptions

        fusion_options = FusionOptions(run_config["model_type"])
        fusion_options.__dict__.update(run_config["optimization_options"])
        run_config["optimization_options"] = fusion_options

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        from onnxruntime.transformers import optimizer as transformers_optimizer

        # start with a copy of the config
        run_config = deepcopy(config)
        del (
            run_config["float16"],
            run_config["input_int32"],
            run_config["keep_io_types"],
            run_config["force_fp32_ops"],
        )
        for key in get_external_data_config():
            del run_config[key]

        output_model_path = ONNXModel.resolve_path(os.path.join(output_model_path, os.path.basename(model.model_path)))

        if config["optimization_options"]:
            self._set_fusion_options(run_config)

        optimizer = transformers_optimizer.optimize_model(input=model.model_path, **run_config)

        if config["float16"]:
            op_block_list = config["force_fp32_ops"]
            optimizer.convert_float_to_float16(keep_io_types=config["keep_io_types"], op_block_list=op_block_list)

        if config["input_int32"]:
            optimizer.change_graph_inputs_to_int32()

        # Topologically sort the graph at the end since previous optimizations may have broken it
        optimizer.topological_sort()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(optimizer.model, output_model_path, config, model.name)
