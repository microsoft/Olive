# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import onnx

from olive.passes.onnx.auto_fusion_utils.codegen.ops import is_elementwise_op
from olive.passes.onnx.auto_fusion_utils.utils import DOMAIN, KERNEL_OUTPUT, create_custom_op_name, hash_kernel_info

if TYPE_CHECKING:
    from onnx import NodeProto

    from olive.passes.onnx.utils import OnnxDAG

SHAPE_TYPE = List[Union[str, int]]


class KernelArgs:
    def __init__(self, output_shape: SHAPE_TYPE):
        # output shape of the kernel
        # since there is no reduction, the output shape has the maximum possible dimensions
        self.rank = len(output_shape) or 1
        self.output_shape = self.normalize_symbolic_dims(output_shape or [1], self.rank)

        # mapping from input name to systematic input name
        self.input_map = {}
        self.input_shapes = {}

    @staticmethod
    def normalize_symbolic_dims(shape: SHAPE_TYPE, rank: int) -> SHAPE_TYPE:
        new_shape = []
        for idx, dim in enumerate(shape):
            if isinstance(dim, int):
                new_shape.append(dim)
            else:
                new_shape.append(f"dim_{idx + rank - len(shape)}")
        return new_shape

    def add_input(self, input_name: str, input_shape: SHAPE_TYPE) -> str:
        if input_name in self.input_map:
            # assume shape is the same
            return self.input_map[input_name]

        name = f"input_{len(self.input_map)}"
        self.input_map[input_name] = name
        self.input_shapes[name] = self.normalize_symbolic_dims(input_shape or [1], self.rank)
        return name


class FusionBase(ABC):
    def __init__(self, base_node: str, dag: "OnnxDAG"):
        assert self.is_valid_base_op(base_node, dag), f"Unsupported base op: {base_node}"
        self.dag = dag
        self.base_node = base_node
        self.fused_nodes = []

        self.network = [(base_node, dag.get_input_names_or_scalar(base_node))]

    def __len__(self):
        return len(self.fused_nodes) + 1

    @staticmethod
    def dtype_is_valid(node: str, dag: "OnnxDAG") -> bool:
        # TODO(jambayk): support more dtypes
        return all(dtype == "float32" for dtype in dag.get_input_dtypes(node) + dag.get_output_dtypes(node))

    @staticmethod
    @abstractmethod
    def support_multidirectional_broadcasting() -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_valid_base_op(cls, node: str, dag: "OnnxDAG") -> bool:
        raise NotImplementedError

    def still_exists(self) -> bool:
        return all(node in self.dag.nodes for node in [self.base_node, *self.fused_nodes])

    @property
    def final_node(self) -> str:
        return self.fused_nodes[-1] if self.fused_nodes else self.base_node

    @property
    def output_name(self) -> str:
        return self.dag.get_output_names(self.final_node)[0]

    @property
    def output_shape(self) -> SHAPE_TYPE:
        return self.dag.get_output_shapes(self.final_node)[0]

    @staticmethod
    def is_broadcastable(shape1: SHAPE_TYPE, shape2: SHAPE_TYPE, multidirectional: bool = False) -> bool:
        # assume the symbolic dims in same index are the same
        max_len = max(len(shape1), len(shape2))
        # remove normalize symbolic dims since the ort shape inference has correct symbolic dims
        # shape1 = KernelArgs.normalize_symbolic_dims(shape1, max_len)
        # shape2 = KernelArgs.normalize_symbolic_dims(shape2, max_len)
        if multidirectional:
            shape1 = [1] * (max_len - len(shape1)) + shape1
            shape2 = [1] * (max_len - len(shape2)) + shape2
            return all(s1 == s2 or s1 == 1 or s2 == 1 for s1, s2 in zip(shape1, shape2))
        else:
            if len(shape2) > len(shape1):
                return False
            return all(s2 in (s1, 1) for s1, s2 in zip(shape1[-len(shape2) :], shape2))

    def can_fuse_more(self) -> bool:
        # check if the final node has only one consumer
        return len(self.dag.get_consumers(self.final_node)) == 1

    def is_valid_fusion(self, op_type: str, shape: SHAPE_TYPE) -> bool:
        if not self.can_fuse_more():
            # cannot fuse more nodes
            # final node has multiple consumers
            return False

        if not is_elementwise_op(op_type):
            # only elementwise ops are supported currently
            return False

        if shape and not self.is_broadcastable(
            self.output_shape, shape, multidirectional=self.support_multidirectional_broadcasting()
        ):
            # check if the shapes are broadcastable
            return False

        return True

    def add_fused_node(self, node: str):
        assert self.dtype_is_valid(node, self.dag), f"Invalid dtype: {node}"

        inputs = self.dag.get_input_names_or_scalar(node)
        assert self.output_name in inputs, "Output of current fusion must be input to the new fusion"

        # which input is the output of the current fusion
        input_idx = inputs.index(self.output_name)
        shape = None
        if len(inputs) == 2:
            # shape of the other input
            shape = self.dag.get_input_shapes(node)[1 - input_idx]

        assert self.is_valid_fusion(self.dag.get_op_type(node), shape), f"Invalid fusion: {node}"

        self.fused_nodes.append(node)
        inputs[input_idx] = KERNEL_OUTPUT
        self.network.append((node, inputs))

    def concat_fusion(self, fusion: "FusionBase"):
        assert self.dtype_is_valid(fusion.base_node, fusion.dag), f"Invalid dtype: {fusion.base_node}"

        inputs = fusion.dag.get_input_names_or_scalar(fusion.base_node)
        assert self.output_name in inputs, "Output of current fusion must be input to the new fusion"

        assert self.is_valid_fusion(
            fusion.dag.get_op_type(fusion.base_node), fusion.output_shape
        ), f"Invalid fusion: {fusion.base_node}"

        input_idx = inputs.index(self.output_name)
        self.fused_nodes.extend([fusion.base_node, *fusion.fused_nodes])
        inputs[input_idx] = KERNEL_OUTPUT
        self.network.extend([(fusion.base_node, inputs), *fusion.network[1:]])

    def try_fuse_node(self, node: str) -> bool:
        try:
            self.add_fused_node(node)
            return True
        except AssertionError:
            return False

    def try_concat_fusion(self, fusion: "FusionBase") -> bool:
        try:
            self.concat_fusion(fusion)
            return True
        except AssertionError:
            return False

    def kernel_info(self) -> str:
        kernel_args = KernelArgs(self.output_shape)

        network = []
        for node, inputs in self.network:
            op_type = self.dag.get_op_type(node)
            input_names = [
                (
                    name
                    if (isinstance(name, (int, float)) or name == KERNEL_OUTPUT)
                    else kernel_args.add_input(name, self.dag.get_shape(name))
                )
                for name in inputs
            ]
            network.append([op_type, input_names])

        return {
            "ops": self.get_op_types(),
            "network": network,
            "shapes": kernel_args.input_shapes,
            "output_shape": kernel_args.output_shape,
        }

    def get_hash(self) -> str:
        return hash_kernel_info(self.kernel_info())

    def get_op_types(self) -> List[str]:
        return [self.dag.get_op_type(node) for node in [self.base_node, *self.fused_nodes]]

    def get_node_names(self) -> List[str]:
        return [self.base_node, *self.fused_nodes]

    def fuse(self) -> Tuple["NodeProto", Dict]:
        if not self.fused_nodes:
            # no fusion
            return None, None

        seen_inputs = set()
        node_names = []
        inputs = []
        # TODO(jambayk): handle attributes when supported
        for node, node_inputs in self.network:
            node_names.append(node)
            for input_name in node_inputs:
                if isinstance(input_name, (int, float)) or input_name == KERNEL_OUTPUT or input_name in seen_inputs:
                    continue
                inputs.append(input_name)
                seen_inputs.add(input_name)
        # we only have single output fusion for now
        outputs = [self.output_name]

        kernel_info = self.kernel_info()

        # create a new node
        new_node = onnx.helper.make_node(
            create_custom_op_name(kernel_info),
            inputs=inputs,
            outputs=outputs,
            name="->".join(node_names),
            domain=DOMAIN,
        )
        return new_node, kernel_info


class ElementwiseFusion(FusionBase):
    @staticmethod
    def support_multidirectional_broadcasting() -> bool:
        return True

    @classmethod
    def is_valid_base_op(cls, node: str, dag: "OnnxDAG") -> bool:
        return is_elementwise_op(dag.get_op_type(node)) and cls.dtype_is_valid(node, dag)


class MatMulFusion(FusionBase):
    @staticmethod
    def support_multidirectional_broadcasting() -> bool:
        return False

    @classmethod
    def is_valid_base_op(cls, node: str, dag: "OnnxDAG") -> bool:
        if not cls.dtype_is_valid(node, dag):
            return False

        if dag.get_op_type(node) != "MatMul":
            return False

        # check that the second input is 2D
        input_shapes = dag.get_input_shapes(node)
        return len(input_shapes[1]) == 2


def get_fusion_class(node: str, dag: "OnnxDAG"):
    if ElementwiseFusion.is_valid_base_op(node, dag):
        return ElementwiseFusion
    # disabling MatMul fusion for now since it is not efficient
    # if MatMulFusion.is_valid_base_op(node, dag):
    #     return MatMulFusion
    return None
