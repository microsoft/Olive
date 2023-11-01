# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, List

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class OrtMixedPrecision(Pass):
    """Convert model to mixed precision."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "op_block_list": PassConfigParam(
                type_=List[str],
                default_value=["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "Relu", "Add"],
                description="List of op types to leave as float32",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        """Convert model to mixed precision.

        It detects whether original model has fp16 precision weights,
        and set parameters for float16 conversion automatically.
        """
        from onnxruntime.transformers.float16 import float_to_float16_max_diff
        from onnxruntime.transformers.onnx_model import OnnxModel as OrtOnnxModel

        ort_onnx_model = OrtOnnxModel(model.load_model())

        op_block_list = config["op_block_list"]
        op_full_set = {node.op_type for node in ort_onnx_model.nodes()}
        fp32_op_set = set(op_block_list)
        fp16_op_set = op_full_set.difference(fp32_op_set)
        logger.debug(f"fp32 op: {fp32_op_set} fp16 op: {fp16_op_set}")

        # logits is the first output
        logits_output_name = ort_onnx_model.graph().output[0].name

        # We use the weight in last MatMul node to detect
        # whether the model is stored with float16 weights from training.
        is_weight_fp16_precision = False
        output_name_to_node = ort_onnx_model.output_name_to_node()
        assert logits_output_name in output_name_to_node
        node = output_name_to_node[logits_output_name]
        last_matmul_node = None
        if node.op_type == "MatMul":
            last_matmul_node = node
            logger.debug(f"Found last MatMul node for logits: {node.name}")
            initializer = None
            for node_input in node.input:
                initializer = ort_onnx_model.get_initializer(node_input)
                if initializer is not None:
                    break

            # when the max difference of value after converting float to float16 is lower than a threshold (1e-6),
            # we can deduce that the weights are stored in float16 precision.
            max_diff = float_to_float16_max_diff(initializer)
            logger.debug(f"max diff of converting weights in last MatMul node {node.name}: {max_diff}")
            is_weight_fp16_precision = max_diff < 1e-6
        else:
            logger.warning(f"Failed to find MatMul node for logits. Found {node.op_type} of node {node.name}")

        keep_io_types = []
        node_block_list = []
        if (not is_weight_fp16_precision) and (last_matmul_node is not None):
            # When original weight is float32 precision,
            # keep logits and last MatMul in float32 could get better precision.
            keep_io_types = [logits_output_name]
            node_block_list = [last_matmul_node.name]

        parameters = {
            "keep_io_types": keep_io_types,
            "op_block_list": list(op_block_list),
            "node_block_list": node_block_list,
            "force_fp16_initializers": is_weight_fp16_precision,
        }

        logger.debug(f"auto_mixed_precision parameters: {parameters}")
        ort_onnx_model.convert_float_to_float16(use_symbolic_shape_infer=True, **parameters)
        # topological sort model since the order of nodes may be changed
        ort_onnx_model.topological_sort()

        output_model_path = ONNXModel.resolve_path(output_model_path)
        return model_proto_to_olive_model(ort_onnx_model.model, output_model_path, config)
