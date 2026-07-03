# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import IdentityEliminationPass, RemoveUnusedNodesPass, TopologicalSortPass
from onnx_ir.traversal import RecursiveGraphIterator

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MatMulNBitsToQDQ(Pass):
    """Convert ONNX MatMulNBits nodes to standard ONNX quantized-dequantized (QDQ) format."""

    INT_ELEM_TYPE_MAP: ClassVar[dict] = {
        # int2 and int4 require onnx 1.20+
        (2, True): 26,
        (2, False): 25,
        (4, True): onnx.TensorProto.INT4,
        (4, False): onnx.TensorProto.UINT4,
        (8, True): onnx.TensorProto.INT8,
        (8, False): onnx.TensorProto.UINT8,
    }

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
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
                    " To be deprecated in future versions in favor of use_signed_int."
                ),
            ),
            "use_signed_int": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use signed int data type for the quantized weight. Default is False and uses unsigned"
                    " int data type. Supersedes use_int4 when set to True."
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
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description=(
                    "List of node names to exclude from the conversion. The node names should be the names of the"
                    " MatMulNBits nodes. Default is None."
                ),
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # load the model into the ONNX IR
        ir_model = ir.load(model.model_path)
        # remove unnecessary identity nodes
        IdentityEliminationPass()(ir_model)

        # set of nodes to exclude from the conversion
        nodes_to_exclude = set(config.nodes_to_exclude or [])

        # keep track of existing node names to generate unique new names
        existing_node_names = {node.name for node in RecursiveGraphIterator(ir_model.graph) if node.name}

        num_modified = 0
        two_bit_present = False
        for node in list(RecursiveGraphIterator(ir_model.graph)):
            if node.op_type != "MatMulNBits" or node.name in nodes_to_exclude:
                continue

            graph = node.graph
            graph_inputs = set(graph.inputs)
            node_inputs = list(node.inputs)
            # only deal with the constant matmul case for now
            if not all(
                value is not None and value.const_value is not None and value not in graph_inputs
                for value in node_inputs[1:]
            ):
                continue

            # original output value
            node_output = node.outputs[0]
            is_model_output = node_output in graph.outputs
            output_index = list(graph.outputs).index(node_output) if is_model_output else None

            K = node.attributes["K"].value  # noqa: N806
            N = node.attributes["N"].value  # noqa: N806
            block_size = node.attributes["block_size"].value
            num_k_blocks = math.ceil(K / block_size)
            # will make this a per-axis DQ if num_k_blocks == 1
            # - originally per-axis K == block_size
            # - originally blockwise but K <= block_size
            is_per_axis = num_k_blocks == 1

            bits = node.attributes["bits"].value
            if bits not in [2, 4, 8]:
                logger.debug("%s uses %d bits, only 2, 4 or 8 bits is supported", node.name, bits)
                continue

            # we can only deal with trivial g_idx, dequantize linear does not support g_idx
            if len(node_inputs) >= 5 and node_inputs[4] is not None:
                g_idx = node_inputs[4].const_value.numpy()
                trivial_g_idx = np.arange(K, dtype=np.int32) // block_size
                if not np.array_equal(g_idx, trivial_g_idx):
                    continue

            unsigned_midpoint = 1 << (bits - 1)
            use_signed_int = config.use_signed_int or (bits == 4 and config.use_int4)

            # name for the DQ node
            dq_name = self._get_new_node_name(existing_node_names, node.name, "DequantizeLinear")
            # weight, scales, zeros
            # (value, new_name, unpacked column size)
            quant_inputs = [
                (node_inputs[1], f"{dq_name}.qweight", K),
                (node_inputs[2], f"{dq_name}.scales", num_k_blocks),
            ]
            if len(node_inputs) >= 4 and node_inputs[3] is not None:
                quant_inputs.append((node_inputs[3], f"{dq_name}.qzeros", num_k_blocks))
            dq_inputs = []

            for qi_value, new_qi_name, unpacked_col_size in quant_inputs:
                # get the np array
                # weight: uint8, scales: float32, zeros: uint8
                qi = qi_value.const_value.numpy()
                # reshape to 2D
                qi = qi.reshape(N, -1)

                # there are cases where unpack and repack is not needed: no transpose + no padding
                # but will still do it for simplicity
                if qi.dtype == np.uint8:
                    qi = self._maybe_unpack_on_row(qi, bits)
                    # remove padding if any
                    qi = qi[:, :unpacked_col_size]

                # Make 1-D scale or qzero if per-axis
                if new_qi_name.endswith((".scales", ".qzeros")) and is_per_axis:
                    qi = qi.flatten()

                # skip if is a no-op zero point, DQ zero point is all 0s == unsigned_midpoint in mnb and signed int
                if (
                    not config.add_zero_point
                    and use_signed_int
                    and new_qi_name.endswith(".qzeros")
                    and np.all(qi == unsigned_midpoint)
                ):
                    continue

                if not config.use_transpose_op:
                    # becomes K X N
                    qi = qi.T

                if qi.dtype == np.uint8:
                    if use_signed_int:
                        # no worries about making signed since the values only use 4/8 bits
                        qi = qi.astype(np.int16)
                        # subtract unsigned_midpoint to make it signed
                        # no worries here again since the values are in the range 0-15/255 and numpy uses 2's complement
                        qi -= unsigned_midpoint

                    # pack in the format expected by onnx and create the tensor
                    tensor = onnx.helper.make_tensor(
                        new_qi_name,
                        self.INT_ELEM_TYPE_MAP[(bits, use_signed_int)],
                        qi.shape,
                        self._maybe_pack_on_flat(qi, bits, use_signed_int).tobytes(),
                        raw=True,
                    )
                else:
                    tensor = onnx.numpy_helper.from_array(qi, name=new_qi_name)

                # add the initializer and record its value
                dq_inputs.append(self._register_initializer(graph, tensor))
            # DQ default zp is 0 but MatMulNBits is 8/128, so we need to add a zero tensor with all 8/128s
            # no need to add for int4/int8 if add_zero_point is False
            if len(dq_inputs) == 2 and (config.add_zero_point or not use_signed_int):
                zp_name = f"{dq_name}.qzeros"
                zp_shape = [N] if is_per_axis else ([N, num_k_blocks] if config.use_transpose_op else [num_k_blocks, N])
                zp_tensor = onnx.helper.make_tensor(
                    zp_name,
                    self.INT_ELEM_TYPE_MAP[(bits, use_signed_int)],
                    zp_shape,
                    # no zp in matmulnbits is equivalent to 8/128 uint4/uint8 and 0 int4/int8 in DQ
                    self._maybe_pack_on_flat(
                        np.zeros(N * num_k_blocks, dtype=np.int16) + (0 if use_signed_int else unsigned_midpoint),
                        bits,
                        use_signed_int,
                    ).tobytes(),
                    raw=True,
                )
                dq_inputs.append(self._register_initializer(graph, zp_tensor))

            # onnx dtype for the float tensors (scale, dequantized weight, matmul inputs+outputs)
            float_dtype = ir.DataType(onnx.helper.np_dtype_to_tensor_dtype(node_inputs[2].const_value.numpy().dtype))

            # new nodes to add to the graph
            # ensure that the node names and output names are unique
            # will add the new nodes, make consumers use the new output and remove the node
            # if output is a model output, rename it back to the original name
            new_nodes = []

            # DequantizeLinear
            dq_name = self._get_new_node_name(existing_node_names, node.name, "DequantizeLinear")
            dq_attributes = [
                # for some reason block_wise and per-axis appear to use swapped axis
                # flip the axis if it is per-axis
                ir.AttrInt64("axis", (1 if config.use_transpose_op else 0) ^ (1 if is_per_axis else 0)),
            ]
            if not is_per_axis:
                dq_attributes.append(ir.AttrInt64("block_size", block_size))
            dq_node = ir.Node(
                "", "DequantizeLinear", inputs=dq_inputs, attributes=dq_attributes, num_outputs=1, name=dq_name
            )
            dq_output = dq_node.outputs[0]
            dq_output.name = f"{dq_name}/output_0"
            dq_output.type = ir.TensorType(float_dtype)
            dq_output.shape = ir.Shape([N, K] if config.use_transpose_op else [K, N])
            new_nodes.append(dq_node)

            if config.use_transpose_op:
                # Transpose
                transpose_name = self._get_new_node_name(existing_node_names, node.name, "Transpose")
                transpose_node = ir.Node(
                    "",
                    "Transpose",
                    inputs=[dq_output],
                    attributes=[ir.AttrInt64s("perm", [1, 0])],
                    num_outputs=1,
                    name=transpose_name,
                )
                transpose_output = transpose_node.outputs[0]
                transpose_output.name = f"{transpose_name}/output_0"
                transpose_output.type = ir.TensorType(float_dtype)
                transpose_output.shape = ir.Shape([K, N])
                new_nodes.append(transpose_node)
                matmul_input = transpose_output
            else:
                matmul_input = dq_output

            # MatMul
            matmul_name = self._get_new_node_name(existing_node_names, node.name, "MatMul")
            matmul_node = ir.Node("", "MatMul", inputs=[node_inputs[0], matmul_input], num_outputs=1, name=matmul_name)
            matmul_output = matmul_node.outputs[0]
            matmul_output.name = f"{matmul_name}/output_0"
            # the output shape is the same as the original MatMulNBits node
            matmul_output.type = node_output.type
            matmul_output.shape = node_output.shape
            new_nodes.append(matmul_node)
            final_output = matmul_output

            if len(node_inputs) >= 6 and node_inputs[5] is not None:
                # Bias Add
                # it has bias
                bias_value = node_inputs[5]
                new_bias_i_name = bias_value.name.replace("MatMulNBits", "MatMul")
                bias_initializer = onnx.numpy_helper.from_array(bias_value.const_value.numpy(), name=new_bias_i_name)
                bias_input = self._register_initializer(graph, bias_initializer)

                bias_name = self._get_new_node_name(existing_node_names, node.name, "Add")
                bias_node = ir.Node("", "Add", inputs=[matmul_output, bias_input], num_outputs=1, name=bias_name)
                bias_output = bias_node.outputs[0]
                bias_output.name = f"{bias_name}/output_0"
                # the output shape is the same as the original MatMulNBits node
                bias_output.type = node_output.type
                bias_output.shape = node_output.shape
                new_nodes.append(bias_node)
                final_output = bias_output

            for new_node in new_nodes:
                graph.append(new_node)

            # change the input of the consumers to the new output
            for usage in list(node_output.uses()):
                usage.node.replace_input_with(usage.idx, final_output)

            # remove the original node
            graph.remove(node)

            # rename to original name if it is a model output
            if is_model_output:
                final_output.name = node_output.name
                graph.outputs[output_index] = final_output

            num_modified += 1
            two_bit_present |= bits == 2

        if num_modified == 0:
            logger.info("No MatMulNBits nodes found. Returning the original model.")
            return model

        # remove the now unused quantized initializers and sort the graph
        RemoveUnusedNodesPass()(ir_model)
        TopologicalSortPass()(ir_model)
        logger.debug("Modified %d MatMulNBits nodes", num_modified)
        # this might not work for all models but will just update the opset version to 21
        # if there is an issue, try the logic in OnnxOpVersionConversion
        opset = 25 if two_bit_present else 21
        ir_model.opset_imports[""] = max(opset, ir_model.opset_imports[""])

        # save the model to the output path and return the model
        return model_proto_to_olive_model(ir.to_proto(ir_model), output_model_path, config)

    @staticmethod
    def _register_initializer(graph: ir.Graph, tensor: onnx.TensorProto) -> ir.Value:
        """Register an ONNX initializer tensor on the graph and return its IR value."""
        const_value = ir.from_proto(tensor)
        value = ir.Value(
            name=tensor.name,
            const_value=const_value,
            type=ir.TensorType(const_value.dtype),
            shape=const_value.shape,
        )
        graph.register_initializer(value)
        return value

    @staticmethod
    def _get_new_node_name(existing_node_names: set, old_name: str, op_type: str) -> str:
        """Get a new unique node name based on the old name and the new op type."""
        new_name_base = None
        for suffix in ["MatMulNBits", "MatMul_Q4", "MatMul_Q8"]:
            if old_name and suffix in old_name:
                new_name_base = old_name.replace(suffix, op_type)
                break
        if new_name_base is None:
            new_name_base = f"{old_name}_{op_type}"

        # will add an index if the name already exists
        idx = 0
        new_name = new_name_base
        while new_name in existing_node_names:
            new_name = f"{new_name_base}_{idx}"
            idx += 1

        existing_node_names.add(new_name)
        return new_name

    @staticmethod
    def _maybe_unpack_on_row(tensor: "NDArray", bits: int) -> "NDArray":
        """Unpack uint8 into multiple sub-byte values (stored in uint8) column wise if bits < 8.

        For example:
        - bits=2: unpack 1 uint8 into 4 uint2 values
        - bits=4: unpack 1 uint8 into 2 uint4 values
        - bits=8: no unpacking needed
        """
        if bits == 8:
            # no need to unpack if bits is 8
            return tensor

        # number of sub-byte values packed into one uint8
        num_packed = 8 // bits
        # shifts for extracting each sub-byte value
        # e.g., bits=2: [0, 2, 4, 6], bits=4: [0, 4]
        shifts = np.arange(num_packed, dtype=np.int32) * bits

        # unpack the uint8
        tensor = np.right_shift(tensor[:, :, None], shifts[None, None, :]).astype(np.uint8)
        # mask out to keep only the relevant bits
        mask = (1 << bits) - 1  # e.g., bits=2: 0x3, bits=4: 0xF
        tensor &= mask
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def _maybe_pack_on_flat(tensor: "NDArray", bits: int, signed: bool) -> "NDArray":
        """Pack multiple sub-byte values into uint8/int8 on a flattened tensor if bits < 8.

        For example:
        - bits=2: pack 4 uint2/int2 values into one uint8/int8
        - bits=4: pack 2 uint4/int4 values into one uint8/int8
        - bits=8: cast to int8 or uint8 based on signed
        """
        tensor = tensor.flatten().astype(np.int8 if signed else np.uint8)
        if bits == 8:
            # no need to pack if bits is 8
            return tensor

        # number of sub-byte values to pack into one uint8
        num_packed = 8 // bits

        # pad if necessary to make length divisible by num_packed
        if len(tensor) % num_packed:
            pad_size = num_packed - (len(tensor) % num_packed)
            tensor = np.pad(tensor, (0, pad_size), mode="constant")

        # mask to keep only the relevant bits
        mask = (1 << bits) - 1  # e.g., bits=2: 0x3, bits=4: 0xF

        # pack values: combine num_packed values into one uint8
        # e.g., bits=2: val[0] | (val[1] << 2) | (val[2] << 4) | (val[3] << 6)
        # e.g., bits=4: val[0] | (val[1] << 4)
        result = tensor[0::num_packed] & mask
        for i in range(1, num_packed):
            result |= (tensor[i::num_packed] & mask) << (i * bits)

        return result
