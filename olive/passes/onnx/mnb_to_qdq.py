# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import ml_dtypes
import numpy as np
from onnxscript import ir
from onnxscript.rewriter import pattern as orp

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MatMulNBitsToQDQ(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "use_transpose_op": PassConfigParam(
                type_=bool,
                # TODO(jambayk): decide whether to enable this by default or not
                # CPU-EP: False gives same output on arm Mac/Windows, but not on x64 Linux/Windows
                default_value=False,
                description=(
                    "Whether to use a Transpose operator after the DequantizeLinear operator. If False, the weight"
                    " initializer will be transposed instead. Default is False. True might be more efficient on some"
                    " EPs such as DirectML."
                ),
            ),
            "use_int4": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use int4 data type for the quantized weight. Default is False and uses uint4 data type."
                ),
            ),
            "add_zero_point": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to add zero point for symmetric quantized weights, i.e., DQ zero point is 0. Default is"
                    " False."
                ),
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # 2 Step
        # 1. pattern replacement
        # 2. Repacking
        ir_model = ir.serde.deserialize_model(model.load_model())

        def mat_mul_n_bits_pattern(op, input_A, q_weight, q_scales, q_zeros, g_idx, bias):
            # bias is an optional input
            return op.MatMulNBits(
                input_A,
                q_weight,
                q_scales,
                q_zeros,
                g_idx,
                bias,
                _outputs=["mat_mul_n_bits_out"],  # Bind the output to the name "mat_mul_n_bits_out"
            )

        def _is_initializer(context, value: ir.Value) -> bool:
            graph: ir.Graph = context.graph
            return value in graph.initializers.values()

        def mat_mul_n_bits_pattern_check(context, *, q_weight, g_idx, mat_mul_n_bits_out: ir.Value, **_) -> bool:
            if not _is_initializer(context, q_weight):
                return False
            node: ir.Node = mat_mul_n_bits_out.producer()
            block_size = node.attributes["block_size"].as_int()
            k = node.attributes["K"].as_int()
            if not _is_initializer(g_idx, q_weight):
                return False
            g_idx = g_idx.constant_value.numpy()
            trivial_g_idx = np.arange(k, dtype=np.int32) // block_size
            if not np.array_equal(g_idx, trivial_g_idx):
                # TODO: We can log why the pattern is not matched here
                return False
            return True

        def mat_mul_n_bits_replacement(
            op,
            *,
            input_A: ir.Value,
            q_weight: ir.Value,
            q_scales: ir.Value,
            q_zeros: ir.Value,
            bias: ir.Value,
            mat_mul_n_bits_out: ir.Value,
            **_,
        ):
            node: ir.Node = mat_mul_n_bits_out.producer()
            # TODO(justinchuby): Keep the old name of the node
            k: int = node.attributes["K"].as_int()
            block_size: int = node.attributes["block_size"].as_int()
            num_k_blocks = math.ceil(k / block_size)
            # will make this a per-axis DQ if num_k_blocks == 1
            # - originally per-axis K == block_size
            # - originally blockwise but K <= block_size
            is_per_axis = num_k_blocks == 1

            # DequantizeLinear -> Transpose -> MatMul -> Add (optional)
            dq = op.DequantizeLinear(
                q_weight,
                q_scales,
                q_zeros,
                block_size=None if is_per_axis else block_size,
                # for some reason block_wise and per-axis appear to use swapped axis
                # flip the axis if it is per-axis
                axis=config["use_transpose_op"] or is_per_axis,
            )
            # TODO(justinchuby): Improve the way we mark something that needs repacking
            dq.producer().meta["needs_repacking"] = True
            dq.producer().meta["K"] = k
            dq.producer().meta["N"] = node.attributes["N"].as_int()
            if config["use_transpose_node"]:
                dq = op.Transpose(dq, perm=[1, 0])
            matmul = op.MatMul(input_A, dq)
            if bias is not None:
                matmul = op.Add(matmul, bias)
            return matmul

        replace_mat_mul_n_bits = orp.RewriteRule(
            mat_mul_n_bits_pattern,
            mat_mul_n_bits_pattern_check,
            mat_mul_n_bits_replacement,
        )
        # TODO(justinchuby): Call the rewriter with replace_mat_mul_n_bits

        # 2. Repack the quantized weights
        for node in ir_model.graph:
            if "needs_repacking" not in node.meta:
                continue

            # Add Logic handling input 3

            unpacked_weight_arrays = _unpack_weights(
                node.meta["K"],
                node.meta["N"],
                node.inputs[1].const_value.numpy(),
                node.inputs[2].const_value.numpy(),
                node.inputs[3].const_value.numpy(),
            )
            node.inputs[1].const_value = ir.Tensor(unpacked_weight_arrays[0])
            node.inputs[2].const_value = ir.Tensor(unpacked_weight_arrays[1])
            if len(unpacked_weight_arrays) == 3:
                # TODO(justinchuby): Specify a name to input_3
                input_3 = ir.Value(None)
                input_3.const_value = ir.Tensor(unpacked_weight_arrays[2])
                # TODO(justinchuby): Ensure the node has three inputs
                node.replace_input_with(3, input_3)
                ir_model.graph.register_initializer(input_3)

            # Clear the meta data
            del node.meta["needs_repacking"]
            del node.meta["K"]
            del node.meta["N"]

        # TODO(justinchuby): Register and remove initializers
        ir_model.opset_imports[""] = max(21, ir_model.opset_imports[""])

        return model_proto_to_olive_model(ir.serde.serialize_model(ir_model), output_model_path, config)

    @staticmethod
    def _get_new_node_name(dag: OnnxDAG, old_name: str, op_type: str):
        """Get a new unique node name based on the old name and the new op type."""
        new_name_base = None
        for suffix in ["MatMulNBits", "MatMul_Q4"]:
            if suffix in old_name:
                new_name_base = old_name.replace(suffix, op_type)
                break
        if new_name_base is None:
            new_name_base = f"{old_name}_{op_type}"

        # will add an index if the name already exists
        idx = 0
        new_name = new_name_base
        while dag.has_node(new_name):
            new_name = f"{new_name_base}_{idx}"
            idx += 1

        return new_name

    @staticmethod
    def _unpack_on_row(tensor: "NDArray") -> "NDArray":
        """Unpack uint8 into two uint4 (stored in uint8) column wise."""
        # two uint4 packed into one uint8
        # right 4 bits are the first uint4
        shifts = np.array([0, 4])

        # unpack the uint8
        tensor = np.right_shift(tensor[:, :, None], shifts[None, None, :]).astype(np.uint8)
        # mask out the first 4 bits
        tensor &= 0xF
        return tensor.reshape(tensor.shape[0], -1)
