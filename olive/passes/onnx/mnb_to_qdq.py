# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import onnx

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
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # create a dag from the model
        dag = OnnxDAG.from_model_path(model.model_path)
        # remove unnecessary identity nodes
        dag.remove_identity_nodes()

        num_modified = 0
        for node_name in dag.get_node_names():
            op_type = dag.get_node_op_type(node_name)
            if op_type != "MatMulNBits":
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
                dq_inputs.append(new_qi_name)

                consumers = dag.get_consumers(qi_name)
                if len(consumers) > 1 and not all(dag.get_node_op_type(c) == "MatMulNBits" for c in consumers):
                    # if the initializer is used in multiple nodes, ensure that all of them are MatMulNBits
                    # lets deal with this case later if it arises, could use a unique name if new_qi_name == qi_name
                    continue

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

                if not config["use_transpose_op"]:
                    # becomes K X N
                    qi = qi.T

                if qi.dtype == np.uint8:
                    # pack in the format expected by onnx and create the tensor
                    tensor = onnx.helper.make_tensor(
                        new_qi_name, onnx.TensorProto.UINT4, qi.shape, self._pack_on_flat(qi).tobytes(), raw=True
                    )
                else:
                    tensor = onnx.numpy_helper.from_array(qi, name=new_qi_name)

                # add the initializer
                dag.add_initializer(tensor, graph_idx)
            # DQ default zp is 0 but MatMulNBits is 8, so we need to add a zero tensor with all 8s
            if len(dq_inputs) == 2:
                zp_name = f"{dq_name}.qzeros"
                zp_tensor = onnx.helper.make_tensor(
                    zp_name,
                    onnx.TensorProto.UINT4,
                    [N, num_k_blocks] if config["use_transpose_op"] else [num_k_blocks, N],
                    self._pack_on_flat(np.zeros(N * num_k_blocks, dtype=np.uint8) + 8).tobytes(),
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
                    block_size=block_size,
                    axis=1 if config["use_transpose_op"] else 0,
                )
            )
            new_value_infos.append(
                onnx.helper.make_tensor_value_info(
                    dq_output, float_elem_type, shape=[N, K] if config["use_transpose_op"] else [K, N]
                )
            )

            if config["use_transpose_op"]:
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
            dag.remove_node(node_name)

            # rename to original name if it is a model output
            if is_model_output:
                dag.rename_node_output(final_name, final_output, node_output)
                dag.make_output(node_output)

            num_modified += 1

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
