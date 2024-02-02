# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List, Optional

import onnx
from onnx import NodeProto

from olive.passes.onnx.auto_fusion_utils.codegen.ops import (
    ELEMENTWISE_OPS,
    ELEMENTWISE_TWO_INPUT_OPS,
    is_commutative_op,
)
from olive.passes.onnx.auto_fusion_utils.codegen.ort_generator import create_custom_op
from olive.passes.onnx.auto_fusion_utils.codegen.triton_generator import create_kernel
from olive.passes.onnx.auto_fusion_utils.utils import DOMAIN, create_custom_op_name


class Fusion:
    def __init__(self, dtype: str, base_op: str, fused_ops: Optional[List[str]] = None):
        assert self.is_valid_base_op(base_op), f"Unsupported base op: {base_op}"
        self.base_op = base_op
        self.dtype = dtype
        self.fused_ops = []
        for op in fused_ops or []:
            self.add_fused_op(op)

    @staticmethod
    def is_valid_base_op(op):
        return op == "MatMul" or op in ELEMENTWISE_OPS or op in ELEMENTWISE_TWO_INPUT_OPS

    @staticmethod
    def is_valid_fused_op(op):
        return op in ELEMENTWISE_OPS or op in ELEMENTWISE_TWO_INPUT_OPS

    @staticmethod
    def is_commutative_op(op):
        return is_commutative_op(op)

    def add_fused_op(self, op):
        assert self.is_valid_fused_op(op), f"Unsupported fused op: {op}"
        self.fused_ops.append(op)

    def get_triton_kernel(self) -> Dict:
        return create_kernel(self.base_op, self.fused_ops, self.dtype)

    def get_custom_op_name(self):
        return create_custom_op_name([self.base_op, *self.fused_ops], self.dtype)

    def get_custom_op(self) -> Dict:
        return create_custom_op(self.base_op, self.fused_ops, self.dtype)

    def fuse_nodes(self, nodes: List[NodeProto]) -> NodeProto:
        """Fuse nodes into a single node.

        This assumes that the nodes are compatible for fusion.
        """
        assert [self.base_op, *self.fused_ops] == [
            node.op_type for node in nodes
        ], "Provided node list does not match the fusion pattern"

        inputs = []
        attributes = {}
        for node_idx, node in enumerate(nodes):
            unique_op_name = f"{node.op_type}_{node_idx}".lower()
            if node_idx == 0:
                inputs.extend(node.input)
            else:
                # remove the output of the previous node from the inputs
                # assumes commutative operations if previous output is not the first input
                node_input = list(node.input)
                node_input.remove(nodes[node_idx - 1].output[0])
                inputs.extend(node_input)

            for attribute in node.attribute:
                # TODO(jambayk): handle non scalar attributes
                attributes[f"{unique_op_name}_{attribute.name}"] = attribute.f
        outputs = nodes[-1].output

        return onnx.helper.make_node(
            self.get_custom_op_name(),
            inputs=inputs,
            outputs=outputs,
            name="->".join([node.name for node in nodes]),
            domain=DOMAIN,
            **attributes,
        )
