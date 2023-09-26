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

        op_block_list = config["op_block_list"]
        op_full_set = {node.op_type for node in model.nodes()}
        fp32_op_set = set(op_block_list)
        fp16_op_set = op_full_set.difference(fp32_op_set)
        logger.info(f"fp32 op: {fp32_op_set} fp16 op: {fp16_op_set}")

        # logits is the first output
        logits_output_name = model.get_graph().output[0].name

        # We use the weight in last MatMul node to detect
        # whether the model is stored with float16 weights from training.
        is_weight_fp16_precision = False
        output_name_to_node = model.output_name_to_node()
        assert logits_output_name in output_name_to_node
        node = output_name_to_node[logits_output_name]
        last_matmul_node = None
        if node.op_type == "MatMul":
            last_matmul_node = node
            logger.info(f"Found last MatMul node for logits: {node.name}")
            initializer = None
            for node_input in node.input:
                initializer = model.get_initializer(node_input)
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

        logger.info(f"auto_mixed_precision parameters: {parameters}")
        fp16_model = self._convert_float_to_float16(
            model=model.load_model(), use_symbolic_shape_infer=True, **parameters
        )
        output_model_path = ONNXModel.resolve_path(output_model_path)
        config = self._config_class(**config)
        return model_proto_to_olive_model(fp16_model, output_model_path, config.dict())

    def _convert_float_to_float16(self, model, use_symbolic_shape_infer=True, **kwargs):
        """Convert a model to half (default) or mixed precision.

            To use mixed precision, user need specify which graph inputs, outputs, operator type
            or list of nodes shall keep in float32.

            By default, we use symbolic shape inference to get shape and type information.
            If not, ONNX shape inference will be used.

            Note that symbolic/ONNX shape inference might fail, and the conversion might not proceed
            without shape and type information.

        Args:
            model (ModelProto): ONNX model to be converted to half or mixed precision.
            use_symbolic_shape_infer (bool, optional): use symbolic shape inference instead of onnx shape inference.
                                                   Defaults to True.
            kwargs: other parameters for float_to_float16 conversion. See below for details.
            keep_io_types (Union[bool, List[str]], optional): boolean or a list of float32 input/output names.
                                                              If True, model inputs/outputs should be left as float32.
                                                              Defaults to True.
            op_block_list (List[str], optional): List of operator types to leave as float32.
                                                 Defaults to None, which will use `float16.DEFAULT_OP_BLOCK_LIST`.
            node_block_list (List[str], optional): List of node names to leave as float32. Defaults to None.
            force_fp16_initializers(bool): force converting all float initializers to float16.
                                           Default to false.
            min_positive_val (float, optional): minimal positive value. Defaults to 1e-7.
            max_finite_val (float, optional): maximal finite value. Defaults to 1e4.
        """
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper

        if "keep_io_types" not in kwargs:
            kwargs["keep_io_types"] = True

        if use_symbolic_shape_infer:
            # Use symbolic shape inference since custom operators (like Gelu, SkipLayerNormalization etc)
            # are not recognized by onnx shape inference.
            shape_infer_helper = SymbolicShapeInferenceHelper(model, verbose=0)
            model = shape_infer_helper.infer_shapes(model, auto_merge=True, guess_output_rank=False)

        parameters = {"disable_shape_infer": use_symbolic_shape_infer}
        parameters.update(
            {
                key: kwargs[key]
                for key in [
                    "keep_io_types",
                    "min_positive_val",
                    "max_finite_val",
                    "op_block_list",
                    "node_block_list",
                    "force_fp16_initializers",
                ]
                if key in kwargs
            }
        )

        return convert_float_to_float16(model, **parameters)
