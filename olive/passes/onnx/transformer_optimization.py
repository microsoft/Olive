# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Union

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModel
from olive.model.hf_mappings import HIDDEN_SIZE_NAMES, MODEL_TYPE_MAPPING, NUM_HEADS_NAMES
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Boolean, Categorical

logger = logging.getLogger(__name__)


class OrtTransformersOptimization(Pass):
    """Optimize transformer based models in scenarios where ONNX Runtime does not apply the optimization at load time.
    It is based on onnxruntime.transformers.optimizer."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        from onnxruntime.transformers.fusion_options import FusionOptions

        is_gpu = accelerator_spec.accelerator_type == Device.GPU

        config = {
            "model_type": PassConfigParam(
                type_=str,
                default_value=None,
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
                searchable_values=Categorical([0, 1, 2, 99]),
                description=(
                    "Graph optimization level of Onnx Runtime: "
                    "0 - disable all (default), 1 - basic, 2 - extended, 99 - all."
                ),
            ),
            "use_gpu": PassConfigParam(type_=bool, default_value=is_gpu, description="Flag for GPU inference."),
            "only_onnxruntime": PassConfigParam(
                type_=bool,
                default_value=False,
                searchable_values=Boolean(),
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

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})
        if search_point.get("float16"):
            if accelerator_spec.execution_provider == "TensorrtExecutionProvider":
                logger.info(
                    "TensorRT has its own float16 implementation, please avoid to use float16 in transformers "
                    "optimization. Suggest to set 'trt_fp16_enable' as True in OrtPerfTuning."
                )
                return False
            if accelerator_spec.execution_provider == "CPUExecutionProvider":
                logger.info("CPUExecutionProvider does not support float16 very well, please avoid to use float16.")
                return False
        if search_point.get("use_gpu") and accelerator_spec.execution_provider == "CPUExecutionProvider":
            logger.info("CPUExecutionProvider does not support GPU inference, please avoid to use use_gpu.")
            return False
        if search_point.get("only_onnxruntime") and search_point.get("opt_level") <= 0:
            logger.info("Please specify a positive value for opt_level when only_onnxruntime is True")
            return False
        if (
            search_point.get("opt_level") == 0
            and search_point.get("only_onnxruntime")
            and search_point.get("num_heads") == 0
            and search_point.get("hidden_size") == 0
        ):
            from onnxruntime import __version__ as OrtVersion
            from packaging import version

            if version.parse(OrtVersion) <= version.parse("1.16.0"):
                logger.info(
                    "Ignore this search point because the issue https://github.com/microsoft/onnxruntime/issues/17254"
                )
            return False
        return True

    @staticmethod
    def _set_fusion_options(run_config: Dict[str, Any]):
        from onnxruntime.transformers.fusion_options import FusionOptions

        fusion_options = FusionOptions(run_config["model_type"])
        fusion_options.__dict__.update(run_config["optimization_options"])
        run_config["optimization_options"] = fusion_options

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
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

        if model.model_attributes:
            model_config = model.model_attributes
            input_model_type = model_config.get("model_type", "")
            _model_type = MODEL_TYPE_MAPPING.get(input_model_type, input_model_type)
            run_config["model_type"] = run_config["model_type"] or _model_type
            assert run_config["model_type"] in transformers_optimizer.MODEL_TYPES, (
                f"Unsupported model type: {run_config['model_type']}, please select one from "
                "{transformers_optimizer.MODEL_TYPES} which need to be set under OrtTransformersOptimization.config"
            )
            if run_config["num_heads"] == 0:
                for num_heads_name in NUM_HEADS_NAMES:
                    if num_heads_name in model_config:
                        run_config["num_heads"] = model_config[num_heads_name]
                        break
            if run_config["hidden_size"] == 0:
                for hidden_size_name in HIDDEN_SIZE_NAMES:
                    if hidden_size_name in model_config:
                        run_config["hidden_size"] = model_config[hidden_size_name]
                        break

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
        return model_proto_to_olive_model(optimizer.model, output_model_path, config)
