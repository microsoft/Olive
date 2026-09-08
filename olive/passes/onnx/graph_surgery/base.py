# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from onnxscript import ir

if TYPE_CHECKING:
    from onnx import ModelProto
    from onnxscript.rewriter import pattern


class Surgeon:
    """Base class for surgeons that operate on the ONNX IR model."""

    # Refer to https://microsoft.github.io/onnxscript/intermediate_representation/ir_api.html#onnxscript.ir.Model
    # for the IR model API.

    registry: ClassVar[dict[str, type[Surgeon]]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Surgeon.registry[cls.__name__.lower()] = cls

    def __init__(self):
        pass

    def __call__(self, model: ModelProto) -> ModelProto:
        return ir.to_proto(self.call_ir(ir.from_proto(model)))

    def call_ir(self, model: ir.Model) -> ir.Model:
        # Implement this method in subclasses to operate on the IR model.
        raise NotImplementedError


class ProtoSurgeon(Surgeon):
    """Base class for surgeons that operate on the ONNX model proto directly."""

    def __call__(self, model: ModelProto) -> ModelProto:
        raise NotImplementedError

    def call_ir(self, model: ir.Model) -> ir.Model:
        raise RuntimeError("Implement __call__ method instead of operator on onnx.ModelProto directly.")

    @staticmethod
    def get_node_by_name(model, name: str, match_output: bool = False):
        for node in model.graph.node:
            if (match_output and node.output[0] == name) or (not match_output and node.name == name):
                return node
        return None

    @staticmethod
    def get_tensor_shapes(model) -> dict[str, list[int]]:
        return {info.name: [x.dim_value for x in info.type.tensor_type.shape.dim] for info in model.graph.value_info}

    @staticmethod
    def get_tensor_types(model):
        return {info.name: info.type.tensor_type.elem_type for info in model.graph.value_info}

    @staticmethod
    def get_initializer_types(model):
        return {initializer.name: initializer.data_type for initializer in model.graph.initializer}

    @staticmethod
    def get_initializer_shapes(model) -> dict[str, list[int]]:
        return {initializer.name: initializer.dims for initializer in model.graph.initializer}

    @staticmethod
    def get_initializer_by_name(model, name: str):
        for initializer in model.graph.initializer:
            if initializer.name == name:
                return initializer
        return None

    @staticmethod
    def create_new_name(name: str, old_op: str, new_op: str) -> str:
        return name.replace(old_op, new_op) if old_op in name else f"{name}_{new_op}"


class RewriteRuleSurgeon(Surgeon):
    """Base class for surgeons implemented as onnxscript rewrite rules.

    Subclasses implement :meth:`rules` to return an
    :class:`onnxscript.rewriter.pattern.RewriteRuleSet`, expressing the match
    pattern and its replacement with the ONNX IR op builder. ``call_ir`` applies
    the rules to the IR model in place. Prefer this over manual proto/DAG
    manipulation for local subgraph pattern replacements: the rewriter handles
    operand commutativity, use-count bookkeeping, and dead-node cleanup.
    """

    def rules(self) -> pattern.RewriteRuleSet:
        raise NotImplementedError

    def call_ir(self, model: ir.Model) -> ir.Model:
        self.rules().apply_to_model(model)
        return model
