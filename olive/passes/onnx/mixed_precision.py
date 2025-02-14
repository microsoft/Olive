# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, List, Type

from onnx import ValueInfoProto

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OrtMixedPrecision(Pass):
    """Convert model to mixed precision."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "op_block_list": PassConfigParam(
                type_=List[str],
                default_value=["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "Relu", "Add"],
                description="List of op types to leave as float32",
            ),
            "atol": PassConfigParam(
                type_=float, default_value=1e-6, description="Absolute tolerance for checking float16 conversion"
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        """Convert model to mixed precision.

        It detects whether original model has fp16 precision weights,
        and set parameters for float16 conversion automatically.
        """
        from onnxruntime.transformers.float16 import float_to_float16_max_diff

        op_block_list = config.op_block_list
        op_full_set = {node.op_type for node in model.nodes()}
        fp32_op_set = set(op_block_list)
        fp16_op_set = op_full_set.difference(fp32_op_set)
        logger.info("fp32 op: %s fp16 op: %s", fp32_op_set, fp16_op_set)

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
            logger.info("Found last MatMul node for logits: %s", node.name)
            initializer = None
            for node_input in node.input:
                initializer = model.get_initializer(node_input)
                if initializer is not None:
                    break

            # when the max difference of value after converting float to float16 is lower than a threshold (1e-6),
            # we can deduce that the weights are stored in float16 precision.
            max_diff = float_to_float16_max_diff(initializer)
            logger.debug("max diff of converting weights in last MatMul node %s: %s", node.name, max_diff)
            is_weight_fp16_precision = max_diff < config.atol
        else:
            logger.warning("Failed to find MatMul node for logits. Found %s of node %s", node.op_type, node.name)

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

        logger.info("auto_mixed_precision parameters: %s", parameters)
        fp16_model = self._convert_float_to_float16(
            model=model.load_model(), use_symbolic_shape_infer=True, **parameters
        )
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        return model_proto_to_olive_model(fp16_model, output_model_path, config)

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
            try:
                model_with_shape = shape_infer_helper.infer_shapes(model, auto_merge=True, guess_output_rank=False)

                # auto_merge might cause issue (see https://github.com/microsoft/onnxruntime/issues/15521)
                # we only merge tensor data type but not shape information back to the original onnx model.
                # Note that float16 conversion need data type but not shape information.
                if model_with_shape is not None:
                    name_vi = {}
                    for vi in model_with_shape.graph.value_info:
                        vi_copy = ValueInfoProto()
                        vi_copy.CopyFrom(vi)
                        if hasattr(vi_copy.type, "tensor_type") and hasattr(vi_copy.type.tensor_type, "shape"):
                            vi_copy.type.tensor_type.ClearField("shape")
                        name_vi[vi.name] = vi_copy

                    for vi in model.graph.value_info:
                        if vi.name in name_vi:
                            del name_vi[vi.name]
                    for vi in name_vi.values():
                        model.graph.value_info.append(vi)
            except Exception:
                logger.warning(
                    "Failed to run symbolic shape inference. Please file an issue"
                    " in https://github.com/microsoft/onnxruntime."
                )

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
