# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

import numpy as np
import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MatMulNBitsToQDQ(Pass):
    """Convert ONNX MatMulNBits nodes to standard ONNX quantized-dequantized (QDQ) format."""

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
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # create a dag from the model
        dag = OnnxDAG.from_model_path(model.model_path)
        # remove unnecessary identity nodes
        dag.remove_identity_nodes()

        # if matmulnbits zero point is the following, then the zero point is not needed in the DQ node
        default_mnb_zp = 8 if config.use_int4 else 0
        int_np_dtype = np.int8 if config.use_int4 else np.uint8
        int_elem_type = onnx.TensorProto.INT4 if config.use_int4 else onnx.TensorProto.UINT4

        # set of nodes to exclude from the conversion
        nodes_to_exclude = set(config.nodes_to_exclude or [])

        num_modified = 0
        for node_name in dag.get_node_names():
            op_type = dag.get_node_op_type(node_name)
            if op_type != "MatMulNBits" or node_name in nodes_to_exclude:
                continue

            node_inputs = dag.get_node_inputs(node_name)
            # only deal with the constant matmul case for now
            if not all(dag.is_initializer(i_name) and not dag.is_input(i_name) for i_name in node_inputs[1:]):
                continue

            graph_idx = dag.get_graph_idx(node_name)

            # original output proto
            node_output = dag.get_node_outputs(node_name)[0]
            is_model_output = dag.is_output(node_output)
            node_output_proto = None
            if dag.get_io(node_output).proto:
                node_output_proto = dag.get_io(node_output).proto[-1]
            node_attributes = dag.get_node_attributes(node_name)
            K = node_attributes["K"]  # noqa: N806
            N = node_attributes["N"]  # noqa: N806
            block_size = node_attributes["block_size"]
            num_k_blocks = math.ceil(K / block_size)
            # will make this a per-axis DQ if num_k_blocks == 1
            # - originally per-axis K == block_size
            # - originally blockwise but K <= block_size
            is_per_axis = num_k_blocks == 1

            # only deal with 4 bits (int4) for now
            if node_attributes["bits"] != 4:
                logger.debug("%s uses %d bits, only 4 bits is supported", node_name, node_attributes["bits"])
                continue

            # we can only deal with trivial g_idx, dequantize linear does not support g_idx
            if len(node_inputs) >= 5 and node_inputs[4]:
                g_idx = dag.get_initializer_np_array(node_inputs[4])
                trivial_g_idx = np.arange(K, dtype=np.int32) // block_size
                if not np.array_equal(g_idx, trivial_g_idx):
                    continue

            # name for the DQ node
            dq_name = self._get_new_node_name(dag, node_name, "DequantizeLinear")
            # weight, scales, zeros
            # (name, new_name, unpacked column size)
            quant_inputs = [
                (node_inputs[1], f"{dq_name}.qweight", K),
                (node_inputs[2], f"{dq_name}.scales", num_k_blocks),
            ]
            if len(node_inputs) >= 4 and node_inputs[3]:
                quant_inputs.append((node_inputs[3], f"{dq_name}.qzeros", num_k_blocks))
            dq_inputs = []

            for qi_name, new_qi_name, unpacked_col_size in quant_inputs:
                # get the np array
                # weight: uint8, scales: float32, zeros: uint8
                qi = dag.get_initializer_np_array(qi_name)
                # reshape to 2D
                qi = qi.reshape(N, -1)

                # there are cases where unpack and repack is not needed: no transpose + no padding
                # but will still do it for simplicity
                if qi.dtype == np.uint8:
                    qi = self._unpack_on_row(qi)
                    # remove padding if any
                    qi = qi[:, :unpacked_col_size]

                # Make 1-D scale or qzero if per-axis
                if new_qi_name.endswith((".scales", ".qzeros")) and is_per_axis:
                    qi = qi.flatten()

                # skip if is a no-op zero point
                if not config.add_zero_point and new_qi_name.endswith(".qzeros") and np.all(qi == default_mnb_zp):
                    continue

                if not config.use_transpose_op:
                    # becomes K X N
                    qi = qi.T

                if qi.dtype == np.uint8:
                    if config.use_int4:
                        # no worries about making signed since the values only use 4 bits
                        qi = qi.astype(np.int8)
                        # subtract 8 to make it signed
                        # no worries here again since the values are in the range 0-15 and numpy uses 2's complement
                        qi -= 8

                    # pack in the format expected by onnx and create the tensor
                    tensor = onnx.helper.make_tensor(
                        new_qi_name,
                        int_elem_type,
                        qi.shape,
                        self._pack_on_flat(qi).tobytes(),
                        raw=True,
                    )
                else:
                    tensor = onnx.numpy_helper.from_array(qi, name=new_qi_name)

                # add the initializer
                dag.add_initializer(tensor, graph_idx)
                # add the input name
                dq_inputs.append(new_qi_name)
            # DQ default zp is 0 but MatMulNBits is 8, so we need to add a zero tensor with all 8s
            # no need to add for int4 if add_zero_point is False
            if len(dq_inputs) == 2 and (config.add_zero_point or not config.use_int4):
                zp_name = f"{dq_name}.qzeros"
                zp_shape = [N] if is_per_axis else ([N, num_k_blocks] if config.use_transpose_op else [num_k_blocks, N])
                zp_tensor = onnx.helper.make_tensor(
                    zp_name,
                    int_elem_type,
                    zp_shape,
                    # no zp in matmulnbits is equivalent to 8 uint4 and 0 int4 in DQ
                    self._pack_on_flat(np.zeros(N * num_k_blocks, dtype=int_np_dtype) + 8 - default_mnb_zp).tobytes(),
                    raw=True,
                )
                dag.add_initializer(zp_tensor, graph_idx)
                dq_inputs.append(zp_name)

            # onnx dtype for the float tensors (scale, dequantized weight, matmul inputs+outputs)
            float_elem_type = onnx.helper.np_dtype_to_tensor_dtype(dag.get_initializer_np_array(node_inputs[2]).dtype)

            # new nodes and value infos to add to the graph
            # ensure that the node names and output names are unique
            # will add the new nodes, make consumers use the new output and remove the node
            # if output is a model output, rename it back to the original name
            new_nodes = []
            new_value_infos = []

            # DequantizeLinear
            dq_name = self._get_new_node_name(dag, node_name, "DequantizeLinear")
            dq_output = f"{dq_name}/output_0"
            new_nodes.append(
                onnx.helper.make_node(
                    "DequantizeLinear",
                    dq_inputs,
                    [dq_output],
                    name=dq_name,
                    block_size=None if is_per_axis else block_size,
                    # for some reason block_wise and per-axis appear to use swapped axis
                    # flip the axis if it is per-axis
                    axis=(1 if config.use_transpose_op else 0) ^ (1 if is_per_axis else 0),
                )
            )
            new_value_infos.append(
                onnx.helper.make_tensor_value_info(
                    dq_output, float_elem_type, shape=[N, K] if config.use_transpose_op else [K, N]
                )
            )

            if config.use_transpose_op:
                # Transpose
                transpose_name = self._get_new_node_name(dag, node_name, "Transpose")
                transpose_output = f"{transpose_name}/output_0"
                new_nodes.append(
                    onnx.helper.make_node(
                        "Transpose", [dq_output], [transpose_output], name=transpose_name, perm=[1, 0]
                    )
                )
                new_value_infos.append(
                    onnx.helper.make_tensor_value_info(transpose_output, float_elem_type, shape=[K, N])
                )
                matmul_input = transpose_output
            else:
                matmul_input = dq_output

            # MatMul
            matmul_name = self._get_new_node_name(dag, node_name, "MatMul")
            matmul_output = f"{matmul_name}/output_0"
            new_nodes.append(
                onnx.helper.make_node("MatMul", [node_inputs[0], matmul_input], [matmul_output], name=matmul_name)
            )
            if node_output_proto:
                # the output shape is the same as the original MatMulNBits node
                matmul_output_proto = onnx.ValueInfoProto()
                matmul_output_proto.CopyFrom(node_output_proto)
                matmul_output_proto.name = matmul_output
                new_value_infos.append(matmul_output_proto)
            final_name = matmul_name
            final_output = matmul_output

            if len(node_inputs) >= 5 and node_inputs[4]:
                # Bias Add
                # it has bias
                bias_i_name = node_inputs[4]
                new_bias_i_name = bias_i_name.replace("MatMulNBits", "MatMul")
                bias_initiaizer = onnx.numpy_helper.from_array(
                    dag.get_initializer_np_array(bias_i_name), name=new_bias_i_name
                )
                dag.add_initializer(bias_initiaizer, graph_idx)

                bias_name = self._get_new_node_name(dag, node_name, "Add")
                bias_output = f"{bias_name}/output_0"
                new_nodes.append(
                    onnx.helper.make_node("Add", [matmul_output, new_bias_i_name], [bias_output], name=bias_name)
                )
                if node_output_proto:
                    # the output shape is the same as the original MatMulNBits node
                    bias_output_proto = onnx.ValueInfoProto()
                    bias_output_proto.CopyFrom(node_output_proto)
                    bias_output_proto.name = bias_output
                    new_value_infos.append(bias_output_proto)
                final_name = bias_name
                final_output = bias_output

            for node in new_nodes:
                dag.add_node(node, graph_idx)

            # change the input of the consumers
            for consumer in dag.get_consumers(node_name):
                dag.replace_node_input(consumer, node_output, final_output)

            # add the new value infos
            for vi in new_value_infos:
                dag.add_value_info(vi, graph_idx)

            # remove the node
            if is_model_output:
                dag.remove_output(node_output)
            dag.remove_node(node_name)

            # rename to original name if it is a model output
            if is_model_output:
                dag.rename_node_output(final_name, final_output, node_output)
                dag.make_output(node_output)

            num_modified += 1

        if num_modified == 0:
            logger.info("No MatMulNBits nodes found. Returning the original model.")
            return model

        dag.update()
        logger.debug("Modified %d MatMulNBits nodes", num_modified)
        # this might not work for all models but will just update the opset version to 21
        # if there is an issue, try the logic in OnnxOpVersionConversion
        dag.model.opset_import[0].version = max(21, dag.model.opset_import[0].version)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(dag.model, output_model_path, config)

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

    @staticmethod
    def _pack_on_flat(tensor: "NDArray") -> "NDArray":
        """Pack two uint4 into one uint8 on a flattened tensor."""
        tensor = tensor.flatten()

        if len(tensor) % 2:
            tensor = np.pad(tensor, (0, 1), mode="constant")

        # right 4 bits are the first uint4
        return (tensor[0::2] & 0xF) | ((tensor[1::2] & 0xF) << 4)
