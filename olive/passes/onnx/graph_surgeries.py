# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
from __future__ import annotations

import inspect
import itertools
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import onnx
import onnxscript
from onnx import ModelProto
from onnx.helper import make_tensor
from onnx_ir.passes.common import (
    DeduplicateHashedInitializersPass,
    InlinePass,
    RemoveUnusedOpsetsPass,
    ShapeInferencePass,
    TopologicalSortPass,
)
from onnxscript import ir, rewriter
from onnxscript.rewriter import pattern

from olive.constants import MSFT_DOMAIN, OpType
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from collections.abc import Sequence

    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ONNXModelHandler


logger = logging.getLogger(__name__)


# pylint: disable=W0621


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
    def find_node(dag: OnnxDAG, op_type: str, name_substr: str) -> str | None:
        """Find the first node matching an op_type and name substring."""
        for name in dag.get_node_names():
            if dag.get_node_op_type(name) == op_type and name_substr in name:
                return name
        return None

    @staticmethod
    def create_new_name(name: str, old_op: str, new_op: str) -> str:
        return name.replace(old_op, new_op) if old_op in name else f"{name}_{new_op}"

    @staticmethod
    def add_reshape_node(
        dag: OnnxDAG,
        graph_idx: int,
        node_name: str,
        input_name: str,
        target_shape: str | list[int],
        output_elem_type: int | None = None,
    ) -> str:
        """Add a reshape node to the graph.

        :param dag: The OnnxDAG object.
        :param graph_idx: The index of the graph.
        :param node_name: The name of the node.
        :param input_name: The name of the input tensor.
        :param target_shape: The target shape for the reshape operation. Can be a string for an existing io or a list of integers.
        :param output_elem_type: The element type of the output tensor. If None, value info will not be added.
        :return: The name of the output tensor after reshaping.
        """
        # need to reshape the first input to 2D
        if isinstance(target_shape, str):
            reshape_shape_name = target_shape
        else:
            reshape_shape_name = f"{node_name}_shape"
            dag.add_initializer(
                onnx.numpy_helper.from_array(np.array(target_shape, dtype=np.int64), reshape_shape_name), graph_idx
            )
        reshape_output_name = f"{node_name}_output"

        dag.add_node(
            onnx.helper.make_node(
                "Reshape",
                [input_name, reshape_shape_name],
                [reshape_output_name],
                name=node_name,
            ),
            graph_idx,
        )
        if not isinstance(target_shape, str) and output_elem_type is not None:
            dag.add_value_info(
                onnx.helper.make_tensor_value_info(
                    reshape_output_name,
                    output_elem_type,
                    target_shape,
                ),
                graph_idx,
            )
        return reshape_output_name


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


# TODO(anyone): This is incorrect, remove or fix
class RenameInputs(Surgeon):
    def __init__(self, old_names: list[str], new_names: list[str]):
        self.old_names = old_names
        self.new_names = new_names

    def call_ir(self, model: ir.Model) -> ir.Model:
        replacement = dict(zip(self.old_names, self.new_names))
        for inp in model.graph.inputs:
            if inp.name in replacement:
                inp.name = replacement[inp.name]
        return model


# TODO(anyone): This is incorrect, remove or fix
class RenameOutputs(Surgeon):
    def __init__(self, old_names: list[str], new_names: list[str]):
        self.old_names = old_names
        self.new_names = new_names

    def call_ir(self, model: ir.Model) -> ir.Model:
        replacement = dict(zip(self.old_names, self.new_names))
        for output in model.graph.outputs:
            if output.name in replacement:
                output.name = replacement[output.name]
        return model


class InferShapes(Surgeon):
    def call_ir(self, model: ir.Model) -> ir.Model:
        return ShapeInferencePass()(model).model


class RemoveShapes(Surgeon):
    def call_ir(self, model: ir.Model) -> ir.Model:
        # value_info is emitted for intermediate values that carry a type/shape;
        # clearing those on non-output node results empties graph.value_info.
        graph_outputs = set(model.graph.outputs)
        for node in model.graph:
            for value in node.outputs:
                if value not in graph_outputs:
                    value.shape = None
                    value.type = None
        return model


class RemoveInitializerFromInputs(Surgeon):
    def call_ir(self, model: ir.Model) -> ir.Model:
        while model.graph.inputs and (model.graph.inputs[-1].name in model.graph.initializers):
            # Initializers are always at the end of the input list
            model.graph.inputs.pop()
        return model


class ReorderInputs(Surgeon):
    """Reorder the inputs of the model according to the given permutation."""

    def __init__(self, permutation: Sequence[int]):
        self.permutation = permutation

    def call_ir(self, model: ir.Model) -> ir.Model:
        inputs = list(model.graph.inputs)
        num_inputs = len(inputs)

        if sorted(self.permutation) != list(range(num_inputs)):
            raise ValueError("Invalid permutation: permutation must be a rearrangement of input indices.")

        reordered_inputs = [inputs[idx] for idx in self.permutation]
        model.graph.inputs.clear()
        model.graph.inputs.extend(reordered_inputs)

        return model


class ReplaceErfWithTanh(RewriteRuleSurgeon):
    """Replace ``Erf(x)`` with ``Tanh(x * 605/503)``.

    ``605/503`` is the standard rational approximation that lets ``tanh`` stand in
    for ``erf``. The scale is emitted as an initializer of the input's
    floating-point dtype; non-floating-point inputs are left unchanged.
    """

    # ir.DataType -> numpy dtype for the emitted scale initializer.
    _DTYPE_MAP: ClassVar[dict] = {
        ir.DataType.FLOAT: np.float32,
        ir.DataType.FLOAT16: np.float16,
        ir.DataType.DOUBLE: np.float64,
    }

    def rules(self) -> pattern.RewriteRuleSet:
        # Unique per-match initializer names (one scale per replaced Erf).
        counter = itertools.count()

        def _pattern(op, x):
            return op.Erf(x)

        def _condition(context, x) -> bool:
            return x.dtype in self._DTYPE_MAP

        def _replacement(op, x):
            scale = op.initializer(
                ir.tensor(
                    np.array(605 / 503, dtype=self._DTYPE_MAP[x.dtype]),
                    name=f"erf_tanh_scale_{next(counter)}",
                )
            )
            return op.Tanh(op.Mul(x, scale))

        return pattern.RewriteRuleSet([pattern.RewriteRule(_pattern, _replacement, _condition)])


class ZeroOutInput(Surgeon):
    def __init__(self, node_name, input_idx):
        self.node_name = node_name
        self.input_idx = input_idx

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        node = next((n for n in graph if n.name == self.node_name), None)
        if node is None:
            logger.warning("Node %s not found in the model.", self.node_name)
            return model

        target = node.inputs[self.input_idx]
        if target is None or target.shape is None or target.dtype is None:
            logger.warning("Cannot determine shape and type for input %d of node '%s'.", self.input_idx, self.node_name)
            return model

        # Concrete zero tensor matching the input's type; dynamic dims default to 1.
        dims = [int(d) if isinstance(d, int) else 1 for d in target.shape]
        zeros = np.zeros(dims, dtype=target.dtype.numpy())

        zero_node = ir.Node(
            "",
            "Constant",
            inputs=[],
            attributes=[ir.AttrTensor("value", ir.tensor(zeros))],
            num_outputs=1,
            name=f"{self.node_name}_zero",
        )
        zero_node.outputs[0].name = f"{self.node_name}_zero_output_0"
        graph.append(zero_node)
        node.replace_input_with(self.input_idx, zero_node.outputs[0])
        return model


class RemoveInputs(Surgeon):
    def __init__(self, names):
        self.names = names

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        names = set(self.names)

        # Drop the matching graph inputs.
        for graph_input in list(graph.inputs):
            if graph_input.name in names:
                graph.inputs.remove(graph_input)

        # Drop references to the removed inputs from each node; remove nodes that
        # end up with no inputs (matching the original behavior).
        for node in list(graph):
            kept = [inp for inp in node.inputs if inp is None or inp.name not in names]
            if len(kept) == len(node.inputs):
                continue
            if not kept:
                graph.remove(node, safe=True)
            else:
                for idx, value in enumerate(kept):
                    node.replace_input_with(idx, value)
                node.resize_inputs(len(kept))
        return model


class ExposeOutputs(Surgeon):
    def __init__(self, names: Sequence[str]):
        self.names = set(names)

    def call_ir(self, model: ir.Model) -> ir.Model:
        for node in model.graph:
            if node.name in self.names:
                model.graph.outputs.append(node.outputs[0])
        return model


class ExposeQuantizedOutput(ProtoSurgeon):
    def __init__(self, output_name):
        self.output_name = output_name

    def _make_name(self, name):
        return f"{self.output_name}_exposed_{name}"

    def _remove_output(self, model):
        idx = -1
        for i, output in enumerate(model.graph.output):
            if output.name == self.output_name:
                model.graph.output.pop(i)
                idx = i
                break
        if idx == -1:
            raise ValueError(f"Output '{self.output_name}' not found in model outputs.")
        return idx

    def _remove_node(self, model, target):
        idx = -1
        for i, node in enumerate(model.graph.node):
            if node == target:
                model.graph.node.pop(i)
                idx = i
                break
        if idx == -1:
            raise ValueError(f"Node '{target.name}' not found in model nodes.")
        return idx

    def _add_protos_to_model(self, model, initializer, node, tensor_type, tensor_shape):
        model.graph.initializer.append(initializer)
        model.graph.node.append(node)
        value_info = onnx.helper.make_tensor_value_info(name=node.output[0], elem_type=tensor_type, shape=tensor_shape)
        model.graph.output.append(value_info)
        return model

    def _add_scale(self, model, scale_value):
        name = self._make_name("scale")
        scale_array = np.array([scale_value], dtype=np.float32)
        initializer = make_tensor(
            name=f"{name}_value", data_type=onnx.TensorProto.FLOAT, dims=scale_array.shape, vals=scale_array
        )
        node = onnx.helper.make_node(
            op_type="Identity",
            name=name,
            inputs=[initializer.name],
            outputs=[f"{name}_output"],
        )
        tensor_type = onnx.TensorProto.FLOAT
        tensor_shape = scale_array.shape
        return self._add_protos_to_model(model, initializer, node, tensor_type, tensor_shape)

    def _add_zero_point(self, model, zero_point_value, onnx_dtype, np_dtype):
        name = self._make_name("zero_point")
        zero_point_array = np.array([zero_point_value], dtype=np_dtype)
        initializer = make_tensor(
            name=f"{name}_value", data_type=onnx_dtype, dims=zero_point_array.shape, vals=zero_point_array
        )
        node = onnx.helper.make_node(
            op_type="Identity", name=name, inputs=[initializer.name], outputs=[f"{name}_output"]
        )
        tensor_type = onnx_dtype
        tensor_shape = zero_point_array.shape
        return self._add_protos_to_model(model, initializer, node, tensor_type, tensor_shape)

    def __call__(self, model: ModelProto):
        from onnx.helper import tensor_dtype_to_np_dtype
        from onnx.numpy_helper import to_array

        output_idx = self._remove_output(model)

        dequantized_node = self.get_node_by_name(model, self.output_name, match_output=True)
        if dequantized_node is None:
            raise ValueError(f"Dequantized node producing output '{self.output_name}' not found.")

        _ = self._remove_node(model, dequantized_node)

        quantized_tensor_name = dequantized_node.input[0]
        quantized_node = self.get_node_by_name(model, quantized_tensor_name, match_output=True)
        if quantized_node is None:
            raise ValueError(f"Quantized node producing tensor '{quantized_tensor_name}' not found.")

        quantized_output_value_info = onnx.helper.make_tensor_value_info(
            name=quantized_node.output[0], elem_type=onnx.TensorProto.UINT8, shape=[]
        )
        model.graph.output.insert(output_idx, quantized_output_value_info)

        scale_initializer = self.get_initializer_by_name(model, quantized_node.input[1])
        if scale_initializer is None:
            raise ValueError(f"Scale initializer '{quantized_node.input[1]}' not found.")
        scale_value = to_array(scale_initializer)[0]
        model = self._add_scale(model, scale_value)

        zero_point_initializer = self.get_initializer_by_name(model, quantized_node.input[2])
        if zero_point_initializer is None:
            raise ValueError(f"Zero point initializer '{quantized_node.input[2]}' not found.")
        zero_point_value = to_array(zero_point_initializer)[0]
        zero_point_onnx_dtype = zero_point_initializer.data_type
        zero_point_np_dtype = tensor_dtype_to_np_dtype(zero_point_onnx_dtype)
        return self._add_zero_point(model, zero_point_value, zero_point_onnx_dtype, zero_point_np_dtype)


class DecomposeQuickGelu(Surgeon):
    """Lower QuickGelu operator to standard ONNX operators.

    QuickGelu pattern:
    [Input] --> QuickGelu --> [Output]

    Replaced with:
    [Input] --> Mul --> Sigmoid --> Mul --> [Output]
                 |                    ^
                 |                    |
                 +--------------------+

    Where the first Mul multiplies by alpha (default 1.702).
    QuickGelu(x) = x * sigmoid(alpha * x)
    """

    ALPHA = 1.702  # QuickGelu default alpha

    def __init__(self):
        super().__init__()
        self._rule = pattern.RewriteRule(
            self._target_pattern,
            self._replacement_pattern,
        )

    def _target_pattern(self, op, x):
        return op.QuickGelu(x, _domain=MSFT_DOMAIN)

    def _replacement_pattern(self, op, x):
        # Create alpha constant
        x_dtype = x.dtype if x.dtype is not None else ir.DataType.FLOAT
        alpha = op.Constant(value=ir.tensor(self.ALPHA, dtype=x_dtype))

        # Compute: x * sigmoid(alpha * x)
        alpha_x = op.Mul(alpha, x)
        sigmoid_alpha_x = op.Sigmoid(alpha_x)
        return op.Mul(x, sigmoid_alpha_x)

    def call_ir(self, model: ir.Model) -> ir.Model:
        modified_model = rewriter.rewrite(model, pattern_rewrite_rules=[self._rule])
        logger.debug("Applied QuickGelu to Mul->Sigmoid->Mul rewrite rule")
        return modified_model


class DecomposeRotaryEmbedding(Surgeon):
    """Lower RotaryEmbedding operator to standard ONNX operators (RoPE).

    RotaryEmbedding pattern:
    [Input] --> RotaryEmbedding --> [Output]
                     ^    ^
                     |    |
          [position_ids] [cos_cache, sin_cache]

    Replaced with:
                                    [position_ids]
                                         |
                                      Reshape
                                         |
                          +--------------+--------------+
                          |                             |
                          v                             v
                    [cos_cache] --> Gather       [sin_cache] --> Gather
                                      |                             |
                                      v                             v
                                   Reshape                       Reshape
                                      |                             |
    [Input] --> (Scale) --> Split ----+-----------------------------+
                              |       |                             |
                              v       v                             v
                            real    imag                         cos, sin
                              |       |                             |
                              +-------+-----------------------------+
                              |       |       |          |          |
                              v       v       v          v          v
                            Mul     Mul     Mul        Mul      (RoPE)
                              |       |       |          |
                              v       v       v          v
                            Sub <-----+     Add <--------+
                              |               |
                              v               v
                            real'           imag'
                              |               |
                              +-------+-------+
                                      |
                                   Concat
                                      |
                                      v
                                  [Output]

    """

    def __init__(self):
        super().__init__()
        self._rule = pattern.RewriteRule(
            self._target_pattern,
            self._replacement_pattern,
        )

    def _target_pattern(self, op, x, position_ids, cos_cache, sin_cache):
        return op.RotaryEmbedding(
            x, position_ids, cos_cache, sin_cache, _domain=MSFT_DOMAIN, _outputs=["rotaryembedding_out"]
        )

    def _replacement_pattern(self, op, x, position_ids, cos_cache, sin_cache, rotaryembedding_out: ir.Value):
        node = rotaryembedding_out.producer()
        # attrs with defaults
        interleaved = node.attributes.get_int("interleaved", 0)
        num_heads = node.attributes.get_int("num_heads", 0)
        rotary_embedding_dim = node.attributes.get_int("rotary_embedding_dim", 0)
        scale = node.attributes.get_float("scale", 1.0)

        # constants (all 1D)
        minus_one_1d = op.Constant(value=ir.tensor([-1], dtype=ir.DataType.INT64))  # shape [1]
        zero_1d = op.Constant(value=ir.tensor([0], dtype=ir.DataType.INT64))
        one_1d = op.Constant(value=ir.tensor([1], dtype=ir.DataType.INT64))
        two_1d = op.Constant(value=ir.tensor([2], dtype=ir.DataType.INT64))

        # 1) gather cos/sin by position_ids
        pos_flat = op.Reshape(position_ids, minus_one_1d)
        cos_g = op.Gather(cos_cache, pos_flat, axis=0)
        sin_g = op.Gather(sin_cache, pos_flat, axis=0)

        # 2) apply scale if needed
        scaled = x
        if scale != 1.0:
            scale_t = op.Constant(value=ir.tensor(scale, dtype=ir.DataType.FLOAT))
            scaled = op.Mul(x, scale_t)

        # 3) compute reshape dims for cos/sin
        shape = op.Shape(scaled)  # [batch, seq, ...]
        batch_dim = op.Gather(shape, zero_1d, axis=0)
        if num_heads > 0:
            num_heads_dim = op.Gather(shape, one_1d, axis=0)
            seq_dim = op.Gather(shape, op.Constant(value=ir.tensor([2], dtype=ir.DataType.INT64)), axis=0)
        else:
            seq_dim = op.Gather(shape, one_1d, axis=0)

        # compute coverage dim
        if rotary_embedding_dim > 0:
            if interleaved:
                cov_dim = op.Constant(value=ir.tensor([rotary_embedding_dim], dtype=ir.DataType.INT64))
            else:
                cov_dim = op.Constant(value=ir.tensor([rotary_embedding_dim // 2], dtype=ir.DataType.INT64))
        else:
            last = op.Gather(shape, minus_one_1d, axis=0)
            cov_dim = op.Div(last, two_1d)  # still 1D

        # build reshape shape (1D of length 4 or 3)
        if num_heads > 0:
            reshape_dims = op.Concat(batch_dim, num_heads_dim, seq_dim, cov_dim, axis=0)
        else:
            reshape_dims = op.Concat(batch_dim, seq_dim, cov_dim, axis=0)

        cos_r = op.Reshape(cos_g, reshape_dims)
        sin_r = op.Reshape(sin_g, reshape_dims)

        # 4) split real/imag
        last_size = op.Gather(op.Shape(scaled), minus_one_1d, axis=0)  # [hidden_size]
        half_size = op.Div(last_size, two_1d)  # [hidden_size/2]
        axes_1d = minus_one_1d  # [-1] slice on last axis

        if interleaved:
            # produce scalars for Range
            zero_s = op.Squeeze(zero_1d, zero_1d)
            two_s = op.Squeeze(two_1d, zero_1d)
            last_s = op.Squeeze(last_size, zero_1d)
            idx_r = op.Range(zero_s, last_s, two_s)
            one_s = op.Squeeze(one_1d, zero_1d)
            idx_i = op.Range(one_s, last_s, two_s)
            in_r = op.Gather(scaled, idx_r, axis=-1)
            in_i = op.Gather(scaled, idx_i, axis=-1)
        else:
            # both starts/ends/axes are 1D
            in_r = op.Slice(scaled, zero_1d, half_size, axes_1d)
            in_i = op.Slice(scaled, half_size, last_size, axes_1d)

        # 5) apply RoPE
        real_cos = op.Mul(in_r, cos_r)
        imag_sin = op.Mul(in_i, sin_r)
        out_r = op.Sub(real_cos, imag_sin)
        real_sin = op.Mul(in_r, sin_r)
        imag_cos = op.Mul(in_i, cos_r)
        out_i = op.Add(real_sin, imag_cos)

        # 6) stitch back
        if interleaved:
            stacked_r = op.Unsqueeze(out_r, minus_one_1d)
            stacked_i = op.Unsqueeze(out_i, minus_one_1d)
            concat_ri = op.Concat(stacked_r, stacked_i, axis=-1)
            final = op.Reshape(concat_ri, op.Shape(x))
        else:
            final = op.Concat(out_r, out_i, axis=-1)

        return final

    def call_ir(self, model: ir.Model) -> ir.Model:
        mod = rewriter.rewrite(model, pattern_rewrite_rules=[self._rule])
        logger.debug("Applied RotaryEmbedding lowering rewrite rule")
        return mod


class RMSNormToL2Norm(Surgeon):
    """Replace RMSNorm subgraph with L2Norm subgraph.

    RMSNorm pattern:
        +-----------------------------------------------+
        |                                               |
        |                                               v
    [Root] --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul
              (y=2)     (axis=-1)   (B=E-6)

    Also handles the case where [Root] is multiplied with rsqrt leading to Div -> Mul
    instead of dividing by sqrt.

    L2Norm pattern:
    [Root] --> LpNormalization --> Mul
                (p=2, axis=-1)

    The weight of the Mul node is multiplied by sqrt(N) where N is equal to the reduced axis size.
    If the weight is all 1s, it is replaced with a 1D array of sqrt(N).
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        modified = 0
        removed_nodes = set()
        replaced_initializers = set()
        for node in list(graph):
            if node in removed_nodes or node.op_type != "Pow":
                continue

            rmsnorm_nodes = self.get_rmsnorm_nodes(node, graph)
            if not rmsnorm_nodes:
                continue

            # name for the new L2Norm node
            l2norm_node_name = ProtoSurgeon.create_new_name(node.name, "Pow", "L2Norm")
            l2norm_node_output_name = f"{l2norm_node_name}_output_0"

            # create L2Norm node
            l2norm_node = ir.Node(
                "",
                "LpNormalization",
                inputs=[node.inputs[0]],
                attributes=[ir.AttrInt64("axis", -1), ir.AttrInt64("p", 2)],
                num_outputs=1,
                name=l2norm_node_name,
            )
            l2norm_node.outputs[0].name = l2norm_node_output_name

            # Muliply weight by sqrt(N)
            final_node_name = rmsnorm_nodes[-1]
            final_node_children = self.get_consumers(final_node_name.outputs[0], graph)
            if len(final_node_children) != 1 or final_node_children[0].op_type not in ["Cast", "Mul"]:
                logger.debug("RMSNorm Pattern does not end with Cast or Mul. Found %s", final_node_children)
                continue
            final_node_output = final_node_name.outputs[0]
            l2norm_node.outputs[0].shape = final_node_output.shape
            l2norm_node.outputs[0].type = final_node_output.type

            # Get the weight Mul node
            if final_node_children[0].op_type == "Mul":
                rmsnorm_mul_node_name = final_node_children[0]
            else:
                cast_node_children = self.get_consumers(final_node_children[0].outputs[0], graph)
                if len(cast_node_children) != 1 or cast_node_children[0].op_type != "Mul":
                    logger.debug("RMSNorm Pattern does not end with Cast -> Mul. Found %s", cast_node_children)
                    continue
                rmsnorm_mul_node_name = cast_node_children[0]

            # Get the weight Mul node inputs
            rmsnorm_weight = None
            for input_value in rmsnorm_mul_node_name.inputs:
                if input_value is not None and input_value.name in graph.initializers:
                    rmsnorm_weight = input_value
                    break
            if rmsnorm_weight is None:
                logger.debug("RMSNorm Mul node does not have initializer input")
                continue

            # update weight and replace initializer
            if rmsnorm_weight.name not in replaced_initializers:
                # rotated models have all 1s and might share initializer
                # don't want to multiply by sqrt(N) multiple times even though it is fine in all 1s case
                rmsnorm_weight_array = rmsnorm_weight.const_value.numpy()
                sqrt_n = np.sqrt(rmsnorm_weight_array.shape[-1]).astype(rmsnorm_weight_array.dtype)
                if np.all(rmsnorm_weight_array == 1):
                    # this is possible in a quarot/spinquant rotated model
                    # Multiplying by 1D is probably faster
                    rmsnorm_weight_array = np.array([1], dtype=rmsnorm_weight_array.dtype)
                rmsnorm_weight_array = sqrt_n * rmsnorm_weight_array

                rmsnorm_weight.const_value = ir.tensor(rmsnorm_weight_array, name=rmsnorm_weight.name)
                replaced_initializers.add(rmsnorm_weight.name)

            # add and replace nodes
            graph.append(l2norm_node)
            final_node_children[0].replace_input_with(
                final_node_children[0].inputs.index(final_node_output), l2norm_node.outputs[0]
            )
            for rms_node_name in rmsnorm_nodes[::-1]:
                graph.remove(rms_node_name)
                removed_nodes.add(rms_node_name)

            modified += 1

        if modified > 0:
            self.remove_unused_constants(graph)
            TopologicalSortPass()(model)
            logger.debug("Replaced %d RMSNorm nodes with L2Norm nodes", modified)

        return model

    @staticmethod
    def get_consumers(value: ir.Value, graph: ir.Graph) -> list[ir.Node]:
        return [node for node in graph if value in node.inputs]

    @classmethod
    def remove_unused_constants(cls, graph: ir.Graph):
        graph_outputs = set(graph.outputs)
        for node in list(graph):
            if node.op_type == "Constant" and all(
                not cls.get_consumers(output, graph) and output not in graph_outputs for output in node.outputs
            ):
                graph.remove(node)

    @classmethod
    def get_rmsnorm_nodes(cls, pow_node: ir.Node, graph: ir.Graph) -> list[ir.Node] | None:
        # Two possible patterns:
        # x / sqrt(mean(x^2) + epsilon): Pow -> ReduceMean -> Add -> Sqrt -> Div
        # x * 1 / sqrt(mean(x^2) + epsilon): Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul
        pattern = ["Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul"]
        pow_node_input = pow_node.inputs[0]
        current_node = pow_node
        rmsnorm_nodes = [current_node]
        for op_type in pattern[1:]:
            child_nodes = cls.get_consumers(current_node.outputs[0], graph)
            if len(child_nodes) != 1 or child_nodes[0].op_type != op_type:
                return []
            current_node = child_nodes[0]
            rmsnorm_nodes.append(current_node)
            if pow_node_input in current_node.inputs:
                # this can happen either at Div or Mul
                # early stopping if it is Div
                break

        return rmsnorm_nodes if len(rmsnorm_nodes) >= (len(pattern) - 1) else []


class SimplifiedLayerNormToRMSNorm(Surgeon):
    """Replace SimplifiedLayerNormalization or SkipSimplifiedLayerNormalization with an RMSNorm subgraph built from elementwise ops.

        RMS(x) = sqrt(mean(x^2, axis=-1, keepdims=1) + eps)
        y      = (x / RMS(x)) * gamma

    For SkipSimplifiedLayerNormalization, we first do:
        s = input + skip
    and use 's' as x for RMSNorm. If the original node exposes a second output
    (residual sum), we rewire its consumers to 's' to preserve graph behavior.

    IMPORTANT: ReduceMean schema change across opsets:
      - opset < 18: axes is an ATTRIBUTE
      - opset >=18: axes is an INPUT tensor (int64), keepdims remains an attribute.
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        use_axes_input_for_reduce_mean = graph.opset_imports.get("", 13) >= 18
        modified = 0

        for node in list(graph):
            op_type = node.op_type
            if op_type not in {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}:
                continue

            inputs = list(node.inputs)
            outputs = [output for output in node.outputs if output.name]

            if op_type == "SkipSimplifiedLayerNormalization":
                # Expect inputs: [input, skip, gamma]
                if len(inputs) != 3:
                    continue
                root1, root2, gamma = inputs

                # Add(input, skip) => skip_add_out
                skip_add_name = ProtoSurgeon.create_new_name(node.name, op_type, "Add")
                skip_add_out = f"{skip_add_name}_out"
                skip_add_node = ir.Node(
                    "",
                    "Add",
                    inputs=[root1, root2],
                    num_outputs=1,
                    name=skip_add_name,
                )
                skip_add_node.outputs[0].name = skip_add_out
                graph.append(skip_add_node)

                ln_input = skip_add_node.outputs[0]
            else:
                # SimplifiedLayerNormalization: inputs = [x, gamma]
                if len(inputs) != 2:
                    continue
                ln_input, gamma = inputs

            # The original primary output (normalized tensor)
            ln_output = outputs[0]

            ln_np_dtype = inputs[0].dtype.numpy() if inputs[0].dtype is not None else np.float32

            pow_name = ProtoSurgeon.create_new_name(node.name, op_type, "Pow")
            pow_out = f"{pow_name}_out"
            pow_const = self.add_initializer(graph, f"{pow_name}_const", np.array([2.0], dtype=ln_np_dtype))
            pow_node = ir.Node(
                "",
                "Pow",
                inputs=[ln_input, pow_const],
                num_outputs=1,
                name=pow_name,
            )
            pow_node.outputs[0].name = pow_out
            graph.append(pow_node)

            mean_name = ProtoSurgeon.create_new_name(node.name, op_type, "ReduceMean")
            mean_out = f"{mean_name}_out"

            if use_axes_input_for_reduce_mean:
                axes_init = self.add_initializer(graph, f"{mean_name}_axes", np.array([-1], dtype=np.int64))
                mean_node = ir.Node(
                    "",
                    "ReduceMean",
                    inputs=[pow_node.outputs[0], axes_init],
                    attributes=[ir.AttrInt64("keepdims", 1)],
                    num_outputs=1,
                    name=mean_name,
                )
            else:
                # Older schema: axes is an attribute
                mean_node = ir.Node(
                    "",
                    "ReduceMean",
                    inputs=[pow_node.outputs[0]],
                    attributes=[ir.AttrInt64s("axes", [-1]), ir.AttrInt64("keepdims", 1)],
                    num_outputs=1,
                    name=mean_name,
                )
            mean_node.outputs[0].name = mean_out
            graph.append(mean_node)

            eps_value = 1e-06
            add_eps_name = ProtoSurgeon.create_new_name(node.name, op_type, "AddEps")
            add_eps_out = f"{add_eps_name}_out"
            eps_const = self.add_initializer(graph, f"{add_eps_name}_const", np.array([eps_value], dtype=ln_np_dtype))
            add_eps_node = ir.Node(
                "",
                "Add",
                inputs=[mean_node.outputs[0], eps_const],
                num_outputs=1,
                name=add_eps_name,
            )
            add_eps_node.outputs[0].name = add_eps_out
            graph.append(add_eps_node)

            sqrt_name = ProtoSurgeon.create_new_name(node.name, op_type, "Sqrt")
            sqrt_out = f"{sqrt_name}_out"
            sqrt_node = ir.Node(
                "",
                "Sqrt",
                inputs=[add_eps_node.outputs[0]],
                num_outputs=1,
                name=sqrt_name,
            )
            sqrt_node.outputs[0].name = sqrt_out
            graph.append(sqrt_node)

            div_name = ProtoSurgeon.create_new_name(node.name, op_type, "Div")
            div_out = f"{div_name}_out"
            div_node = ir.Node(
                "",
                "Div",
                inputs=[ln_input, sqrt_node.outputs[0]],
                num_outputs=1,
                name=div_name,
            )
            div_node.outputs[0].name = div_out
            graph.append(div_node)

            mul_name = ProtoSurgeon.create_new_name(node.name, op_type, "Mul")
            mul_out = f"{mul_name}_out"
            mul_node = ir.Node(
                "",
                "Mul",
                inputs=[div_node.outputs[0], gamma],
                num_outputs=1,
                name=mul_name,
            )
            mul_node.outputs[0].name = mul_out
            graph.append(mul_node)

            ln_output.replace_all_uses_with(mul_node.outputs[0], replace_graph_outputs=True)

            if op_type == "SkipSimplifiedLayerNormalization" and len(outputs) == 2:
                outputs[1].replace_all_uses_with(skip_add_node.outputs[0], replace_graph_outputs=True)

            graph.remove(node)
            modified += 1

        if modified > 0:
            TopologicalSortPass()(model)
            logger.debug(
                "Replaced %d Simplified/SkipSimplifiedLayerNormalization nodes with RMSNorm subgraphs", modified
            )

        return model

    @staticmethod
    def add_initializer(graph: ir.Graph, name: str, array: np.ndarray) -> ir.Value:
        value = ir.Value(name=name, const_value=ir.tensor(array, name=name))
        graph.initializers[name] = value
        return value


class SimplifiedLayerNormToL2Norm(Surgeon):
    """Replace Skip/SimplifiedLayerNormalization node with L2Norm subgraph.

    SimplifiedLayerNormalization is replaced with:
    [Root] --> LpNormalization --> Mul
               (p=2, axis=-1)

    SkipSimplifiedLayerNormalization is replaced with:
    [Root1] -------> Add
                      |
                      v
    [Root2] --> LpNormalization --> Mul
                (p=2, axis=-1)

    Second input to Mul is the weight of the layer norm multiplied by sqrt(N) where N is equal to the hidden size.
    If the weight is all 1s, it is replaced with a 1D array of sqrt(N).
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        modified = 0
        for node in list(graph):
            op_type = node.op_type
            if op_type not in {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}:
                continue

            inputs = list(node.inputs)
            outputs = [output for output in node.outputs if output.name]
            if len(inputs) != 2 + int(op_type == "SkipSimplifiedLayerNormalization"):
                # SimplifiedLayerNormalization: X, scale supported
                # SkipSimplifiedLayerNormalization: input, skip, gamma supported
                continue
            if len(outputs) > 1 + int(op_type == "SkipSimplifiedLayerNormalization"):
                # SimplifiedLayerNormalization: output supported
                # SkipSimplifiedLayerNormalization: output, input_skip_bias_sum (optional) supported
                continue

            if op_type == "SkipSimplifiedLayerNormalization":
                # SimplifiedLayerNormalization preceded by an Add node
                add_node_name = ProtoSurgeon.create_new_name(node.name, op_type, "Add")
                add_node_output_name = f"{add_node_name}_output_0"
                add_node = ir.Node(
                    "",
                    "Add",
                    inputs=inputs[:2],
                    num_outputs=1,
                    name=add_node_name,
                )
                add_node.outputs[0].name = add_node_output_name
                graph.append(add_node)

                # input_skip_bias_sum is used downstream
                if len(outputs) == 2:
                    add_node.outputs[0].shape = outputs[1].shape
                    add_node.outputs[0].type = outputs[1].type
                    outputs[1].replace_all_uses_with(add_node.outputs[0], replace_graph_outputs=True)

                l2norm_node_input = add_node.outputs[0]
            else:
                l2norm_node_input = inputs[0]

            layernorm_node_output = outputs[0]

            # add L2Norm node
            l2norm_node_name = ProtoSurgeon.create_new_name(node.name, op_type, "L2Norm")
            l2norm_node_output_name = f"{l2norm_node_name}_output_0"
            l2norm_node = ir.Node(
                "",
                "LpNormalization",
                inputs=[l2norm_node_input],
                attributes=[ir.AttrInt64("axis", -1), ir.AttrInt64("p", 2)],
                num_outputs=1,
                name=l2norm_node_name,
            )
            l2norm_node.outputs[0].name = l2norm_node_output_name
            l2norm_node.outputs[0].shape = layernorm_node_output.shape
            l2norm_node.outputs[0].type = layernorm_node_output.type
            graph.append(l2norm_node)

            # add Mul node
            mul_weight = inputs[-1]
            if mul_weight is None or mul_weight.name not in graph.initializers or mul_weight.const_value is None:
                logger.debug("SimplifiedLayerNormalization weight is not an initializer")
                continue
            mul_weight_array = mul_weight.const_value.numpy()
            sqrt_n = np.sqrt(mul_weight_array.shape[-1]).astype(mul_weight_array.dtype)
            if np.all(mul_weight_array == 1):
                # this is possible in a quarot/spinquant rotated model
                # Multiplying by 1D is probably faster
                mul_weight_array = np.array([1], dtype=mul_weight_array.dtype)
            mul_weight_array = sqrt_n * mul_weight_array
            mul_weight.const_value = ir.tensor(mul_weight_array, name=mul_weight.name)

            mul_node_name = ProtoSurgeon.create_new_name(node.name, op_type, "Mul")
            mul_node_output_name = f"{mul_node_name}_output_0"
            mul_node = ir.Node(
                "",
                "Mul",
                inputs=[l2norm_node.outputs[0], mul_weight],
                num_outputs=1,
                name=mul_node_name,
            )
            mul_node.outputs[0].name = mul_node_output_name
            mul_node.outputs[0].shape = layernorm_node_output.shape
            mul_node.outputs[0].type = layernorm_node_output.type
            graph.append(mul_node)

            layernorm_node_output.replace_all_uses_with(mul_node.outputs[0], replace_graph_outputs=True)

            # remove node
            graph.remove(node)

            modified += 1

        if modified > 0:
            TopologicalSortPass()(model)
            logger.debug("Replaced %d Skip/SimplifiedLayerNormalization nodes with L2Norm nodes", modified)

        return model


class PowReduceSumPowDiv2LpNorm(Surgeon):
    """Merge Pow ReduceSum Pow Div pattern to L2Norm.

    Below pattern is replaced with LpNormalization (p=2, axis=-1):
    [Root] --> Pow --> ReduceSum --> Pow -- Div
        |                                    |
        +------------------------------------+
    """

    def find_lp_norm_pattern(self, graph: ir.Graph):
        matches = []
        for node in graph:
            if node.op_type != "Div":
                continue

            div_node = node
            pow2_node = None
            sum_node = None
            pow_half_node = None

            # Find Pow node input to Div (norm)
            if len(div_node.inputs) < 2 or div_node.inputs[1] is None:
                continue
            norm_input = div_node.inputs[1]
            for node2 in graph:
                if node2.outputs and node2.outputs[0] is norm_input and node2.op_type == "Pow":
                    pow_half_node = node2
                    break

            if not pow_half_node:
                continue

            sum_input = pow_half_node.inputs[0]
            for node3 in graph:
                if node3.outputs and node3.outputs[0] is sum_input and node3.op_type == "ReduceSum":
                    sum_node = node3
                    break

            if not sum_node:
                continue

            pow_input = sum_node.inputs[0]
            for node4 in graph:
                if node4.outputs and node4.outputs[0] is pow_input and node4.op_type == "Pow":
                    pow2_node = node4
                    break

            if not pow2_node:
                continue

            # Confirm constant exponents: 2 and 0.5
            def is_const_pow(node, exp_val):
                if node.op_type != "Pow" or len(node.inputs) < 2:
                    return False
                exponent = node.inputs[1]
                if exponent is None or exponent.const_value is None:
                    return False
                return exp_val in [exponent.const_value.numpy().item()]

            if not (is_const_pow(pow2_node, 2.0) and is_const_pow(pow_half_node, 0.5)):
                continue

            # Save matched nodes
            matches.append((div_node, pow_half_node, sum_node, pow2_node))

        return matches

    def replace_with_lp_normalization(self, model: ir.Model):
        graph = model.graph
        matches = self.find_lp_norm_pattern(graph)
        logger.info("Replacing %d instance of the pattern with LpNorm", len(matches))
        for div_node, pow_half_node, sum_node, pow2_node in matches:
            input_value = pow2_node.inputs[0]
            output_value = div_node.outputs[0]

            lp_node = ir.Node(
                "",
                "LpNormalization",
                inputs=[input_value],
                attributes=[ir.AttrInt64("p", 2), ir.AttrInt64("axis", -1)],
                num_outputs=1,
                name=div_node.name + "_lp_norm",
            )
            lp_node.outputs[0].name = output_value.name
            lp_node.outputs[0].shape = output_value.shape
            lp_node.outputs[0].type = output_value.type
            graph.append(lp_node)
            output_value.replace_all_uses_with(lp_node.outputs[0], replace_graph_outputs=True)

            # Remove old nodes
            graph.remove([div_node, pow_half_node, sum_node, pow2_node])

        if matches:
            TopologicalSortPass()(model)
        return model

    def call_ir(self, model: ir.Model) -> ir.Model:
        return self.replace_with_lp_normalization(model)


class MatMulAddToGemm(ProtoSurgeon):
    """Replace MatMul + Add with Gemm.

    Second MatMul input must be a 2D tensor and the other input of the Add node must be a 1D tensor.
    If the first MatMul input is more than 2D and the shapes are static, it is reshaped to 2D before the Gemm
    node and reshaped back to the original shape after the Gemm node.
    If a Relu is present after the Add operation and reshaping is required, the post-reshape will be added
    after the Relu operation.
    """

    def __call__(self, model: ModelProto):
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        try:
            model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        except Exception as e:
            logger.debug("Shape inference failed. Will try to continue without it. Error: %s", e)

        dag = OnnxDAG(model)

        modified = 0
        removed_nodes = set()
        for node_name in dag.get_node_names():
            if node_name in removed_nodes or dag.get_node_op_type(node_name) != "MatMul":
                continue
            matmul_name = node_name
            graph_idx = dag.get_graph_idx(node_name)

            matmul_consumers = dag.get_consumers(node_name)
            if len(matmul_consumers) != 1 or dag.get_node_op_type(matmul_consumers[0]) != "Add":
                continue
            add_name = matmul_consumers[0]

            out = dag.get_node_outputs(add_name)[0]
            out_consumers = dag.get_consumers(add_name)
            if dag.is_output(out):
                continue

            # check matmul input shapes
            matmul_inputs = dag.get_node_inputs(node_name)
            matmul_input_shapes = [dag.get_io_shape(i_name) for i_name in matmul_inputs]
            if len(matmul_input_shapes[1]) != 2:
                continue
            matmul_output = dag.get_node_outputs(node_name)[0]
            matmul_output_shape = dag.get_io_shape(matmul_output)
            elem_type = dag.get_io_elem_type(matmul_output)

            # check add input shapes
            bias_input = None
            for i_name in dag.get_node_inputs(add_name):
                if i_name != matmul_output:
                    bias_input = i_name
                    break
            if bias_input is None or len(dag.get_io_shape(bias_input)) != 1:
                continue

            gemm_name = self.create_new_name(matmul_name, "MatMul", "Gemm")
            gemm_inputs = [*matmul_inputs, bias_input]

            # check if there's a Relu after the Add
            has_relu = len(out_consumers) == 1 and dag.get_node_op_type(out_consumers[0]) == "Relu"
            relu_name = out_consumers[0] if has_relu else None

            matmul_a_shape = matmul_input_shapes[0]
            gemm_output_shape = matmul_output_shape
            reshape_needed = len(matmul_a_shape) != 2
            if reshape_needed:
                # need to reshape the first input to 2D
                # only support static shapes for now, otherwise we need to add shape related ops
                if any(
                    not isinstance(dim_value, int) for dim_value in [*matmul_input_shapes[0], *matmul_input_shapes[1]]
                ):
                    continue

                pre_reshape_name = self.create_new_name(gemm_name, "Gemm", "Reshape_pre")
                gemm_inputs[0] = self.add_reshape_node(
                    dag,
                    graph_idx,
                    pre_reshape_name,
                    matmul_inputs[0],
                    [math.prod(matmul_a_shape[:-1]), matmul_a_shape[-1]],
                    elem_type,
                )
                gemm_output_shape = [math.prod(matmul_output_shape[:-1]), matmul_output_shape[-1]]

            gemm_output_name = f"{gemm_name}_output"
            dag.add_node(
                onnx.helper.make_node(
                    "Gemm",
                    inputs=gemm_inputs,
                    outputs=[gemm_output_name],
                    name=gemm_name,
                    alpha=1.0,
                    beta=1.0,
                    transA=0,
                    transB=0,
                ),
                graph_idx,
            )
            dag.add_value_info(
                onnx.helper.make_tensor_value_info(
                    gemm_output_name,
                    elem_type,
                    gemm_output_shape,
                ),
                graph_idx,
            )
            final_output_name = gemm_output_name

            if reshape_needed:
                if has_relu:
                    out = dag.get_node_outputs(relu_name)[0]
                    out_consumers = dag.get_consumers(relu_name)
                    # Connect Gemm output to Relu input
                    dag.replace_node_input(relu_name, dag.get_node_inputs(relu_name)[0], gemm_output_name)
                    # Update Relu output value info
                    relu_output = dag.get_node_outputs(relu_name)[0]
                    new_relu_vi = onnx.helper.make_tensor_value_info(
                        relu_output,
                        elem_type,
                        gemm_output_shape,
                    )
                    dag.add_value_info(new_relu_vi, graph_idx, overwrite=True)
                    final_output_name = dag.get_node_outputs(relu_name)[0]

                # need to reshape the output to original shape
                post_reshape_name = self.create_new_name(gemm_name, "Gemm", "Reshape_post")
                final_output_name = self.add_reshape_node(
                    dag,
                    graph_idx,
                    post_reshape_name,
                    final_output_name,
                    matmul_output_shape,
                    elem_type,
                )

            # point all of the consumers of the original add/relu to the final output
            for consumer in out_consumers:
                dag.replace_node_input(consumer, out, final_output_name)

            # remove the original add and matmul nodes
            for to_remove in [add_name, matmul_name]:
                dag.remove_node(to_remove)
                removed_nodes.add(to_remove)
            modified += 1

        if modified > 0:
            logger.debug("Replaced %d MatMul + Add nodes with Gemm nodes", modified)

        dag.update()
        return dag.model


class GemmToMatMulAdd(ProtoSurgeon):
    """Replace Gemm with MatMul (+ Add) for INT4 quantization compatibility.

    The INT4 RTN quantizer only recognizes MatMul nodes.  This surgeon converts
    Gemm nodes back to MatMul+Add so that the weight matrices become eligible
    for block-wise quantization.

    Handles transB by transposing constant weights in-place or inserting a
    Transpose node for non-constant weights.  Skips Gemm nodes whose alpha/beta
    are not 1.0 or whose transA is set.
    """

    def __call__(self, model: ModelProto):
        from onnx import helper, numpy_helper

        graph = model.graph
        initializer_map = {init.name: init for init in graph.initializer}
        existing_names = (
            {init.name for init in graph.initializer}
            | {vi.name for vi in graph.input}
            | {vi.name for vi in graph.output}
            | {vi.name for vi in graph.value_info}
        )
        nodes_to_remove = []
        nodes_to_add = []
        gemm_rewrite_idx = 0

        for node in graph.node:
            if node.op_type != "Gemm":
                continue

            alpha = beta = 1.0
            trans_a = trans_b = 0
            for attr in node.attribute:
                if attr.name == "alpha":
                    alpha = attr.f
                elif attr.name == "beta":
                    beta = attr.f
                elif attr.name == "transA":
                    trans_a = attr.i
                elif attr.name == "transB":
                    trans_b = attr.i

            if alpha != 1.0 or beta != 1.0 or trans_a != 0:
                continue

            inp_a, inp_b = node.input[0], node.input[1]
            inp_c = node.input[2] if len(node.input) > 2 else None
            out_y = node.output[0]

            # Derive a stable base name for new nodes/tensors.
            base_name = node.name or out_y or f"gemm_rewrite_{gemm_rewrite_idx}"

            if trans_b:
                if inp_b in initializer_map:
                    # Create a new transposed initializer to avoid mutating
                    # a potentially shared initializer in-place.
                    init = initializer_map[inp_b]
                    w_t = numpy_helper.to_array(init).T.copy()
                    new_name = f"{inp_b}_transposed"
                    suffix = 0
                    while new_name in existing_names:
                        suffix += 1
                        new_name = f"{inp_b}_transposed_{suffix}"
                    new_init = numpy_helper.from_array(w_t, name=new_name)
                    graph.initializer.append(new_init)
                    initializer_map[new_name] = new_init
                    existing_names.add(new_name)
                    matmul_rhs = new_name
                else:
                    transpose_out = f"{base_name}_transpose_B"
                    nodes_to_add.append(
                        helper.make_node(
                            "Transpose", [inp_b], [transpose_out], name=f"{base_name}_Transpose", perm=[1, 0]
                        )
                    )
                    matmul_rhs = transpose_out
            else:
                matmul_rhs = inp_b

            if inp_c:
                matmul_out = f"{base_name}_matmul_out"
                nodes_to_add.append(
                    helper.make_node("MatMul", [inp_a, matmul_rhs], [matmul_out], name=f"{base_name}_MatMul")
                )
                nodes_to_add.append(helper.make_node("Add", [matmul_out, inp_c], [out_y], name=f"{base_name}_Add"))
            else:
                nodes_to_add.append(
                    helper.make_node("MatMul", [inp_a, matmul_rhs], [out_y], name=f"{base_name}_MatMul")
                )

            nodes_to_remove.append(node)
            gemm_rewrite_idx += 1

        for node in nodes_to_remove:
            graph.node.remove(node)
        graph.node.extend(nodes_to_add)

        if nodes_to_remove:
            logger.debug("Replaced %d Gemm nodes with MatMul + Add nodes", len(nodes_to_remove))

        return model


class RemoveRopeMultiCache(ProtoSurgeon):
    """Remove the multi rope cache from the model."""

    def __init__(self, use_large_cache: bool = False):
        self.use_large_cache = use_large_cache

    def __call__(self, model: ModelProto):
        dag = OnnxDAG(model)

        supported_consumer_ops = {"GroupQueryAttention", "RotaryEmbedding"}
        if not supported_consumer_ops.intersection(set(dag.get_node_op_types())):
            return dag.model

        # get the first GQA/RotaryEmbedding node
        first_node = None
        for node in dag.get_node_names():
            op_type = dag.get_node_op_type(node)
            if op_type in supported_consumer_ops:
                first_node = node
                break
        first_node_inputs = dag.get_node_inputs(first_node)

        # check if the GQA node has cos_cache and sin_cache inputs
        # GQA can have 9 inputs (onnxruntime-genai <= 0.9.0) or 11 inputs (onnxruntime-genai > 0.9.0)
        if dag.get_node_op_type(first_node) == "GroupQueryAttention" and len(first_node_inputs) < 9:
            return dag.model

        # check if cos_cache and sin_cache come from an If node
        # cos_cache and sin_cache are at positions 7 and 8
        cache_names = {"cos_cache": first_node_inputs[7], "sin_cache": first_node_inputs[8]}
        for cache_name in cache_names.values():
            if dag.is_input(cache_name) or dag.is_initializer(cache_name):
                return dag.model
            if dag.get_node_op_type(dag.get_producer(cache_name)) != "If":
                return dag.model

        for key, cache_name in cache_names.items():
            cache_to_use = f"{key}_large" if self.use_large_cache else f"{key}_small"
            new_cache_name = f"{key}_single"

            if dag.is_initializer(cache_to_use):
                cache_initializer = onnx.TensorProto()
                cache_initializer.CopyFrom(dag.get_initializer_proto(cache_to_use))
                cache_initializer.name = new_cache_name
            else:
                cache_producer_name = dag.get_producer(cache_to_use)
                assert dag.get_node_op_type(cache_producer_name) == "Constant"
                cache_value = onnx.numpy_helper.to_array(
                    onnx.helper.get_attribute_value(dag.get_node_proto(cache_producer_name).attribute[0])
                )
                cache_initializer = onnx.numpy_helper.from_array(cache_value, new_cache_name)

            dag.add_initializer(cache_initializer, 0)
            for consumer in dag.get_consumers(cache_name):
                dag.replace_node_input(consumer, cache_name, new_cache_name)

        # remove the Greater -> If Nodes
        if_node_name = dag.get_producer(cache_names["cos_cache"])
        greater_node_name = dag.get_parents(if_node_name)[0]
        dag.remove_node(if_node_name)
        dag.remove_node(greater_node_name)

        logger.debug("Removed rope multi cache")

        dag.update()
        return dag.model


class AttentionMaskToSequenceLengths(ProtoSurgeon):
    """Replace attention_mask subgraph in GQA model with past_seq_len and total_seq_len."""

    def __call__(self, model: ModelProto):
        dag = OnnxDAG(model)

        if "GroupQueryAttention" not in dag.get_node_op_types():
            return dag.model

        # get first GQA node
        first_node = None
        for node in dag.get_node_names():
            if dag.get_node_op_type(node) == "GroupQueryAttention":
                first_node = node
                break
        first_node_inputs = dag.get_node_inputs(first_node)

        seq_len_names = {"past_seq_len": first_node_inputs[5], "total_seq_len": first_node_inputs[6]}
        # check that both are not already inputs
        for seq_len_name in seq_len_names.values():
            if dag.is_input(seq_len_name):
                return dag.model

        batch_size, _ = dag.get_io_shape("input_ids")
        seq_len_shapes = {"past_seq_len": [batch_size, 1], "total_seq_len": []}
        for key, seq_len_name in seq_len_names.items():
            input_proto = onnx.helper.make_tensor_value_info(key, onnx.TensorProto.INT32, seq_len_shapes[key])
            dag.add_input(input_proto, 0)
            for node in dag.get_consumers(seq_len_name):
                dag.replace_node_input(node, seq_len_name, key)

        # remove attention mask subgraph
        nodes_to_remove = []
        visit_stack = dag.get_consumers("attention_mask")
        while visit_stack:
            node = visit_stack.pop(0)
            assert not dag.is_output_producer(node), "Unexpected graph structure"
            for consumer in dag.get_consumers(node):
                if dag.get_node_op_type(consumer) == "GroupQueryAttention":
                    for node_o in dag.get_node_outputs(node):
                        assert dag.get_node_inputs(consumer).index(node_o) in {5, 6}, "Rope multi cache not supported"
                    continue
                visit_stack.append(consumer)
            nodes_to_remove.append(node)
        for node in nodes_to_remove[::-1]:
            dag.remove_node(node)

        logger.debug("atttention mask replaced with sequence length inputs")

        dag.update()
        return dag.model


class ReplaceAttentionMaskValue(Surgeon):
    """Replace the value of extended attention mask with a new value.

    This surgery is useful if the default mask value does not quantize well due to numerical instability.
    """

    ALLOWED_CONSUMER_OPS: ClassVar[set[str]] = {"Add", "Mul", "Expand", "Where", "Shape"}

    def __init__(self, threshold: float = -3e30, replacement: float = -1e4):
        self.threshold = threshold
        self.replacement = replacement

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        modified = 0

        # Update Constant / ConstantOfShape nodes whose float value has entries below the threshold.
        for node in graph:
            if node.op_type not in ("Constant", "ConstantOfShape"):
                continue
            attr = node.attributes.get("value")
            if (
                attr is None
                or attr.type != ir.AttributeType.TENSOR
                or attr.value.dtype != ir.DataType.FLOAT
                or not self.valid_consumers(node.outputs[0])
            ):
                continue
            new_value = self.new_tensor_value(attr.value.numpy())
            if new_value is not None:
                node.attributes["value"] = ir.AttrTensor("value", ir.tensor(new_value))
                modified += 1

        # Update float initializers with entries below the threshold.
        for value in list(graph.initializers.values()):
            if (
                value.const_value is None
                or value.const_value.dtype != ir.DataType.FLOAT
                or not self.valid_consumers(value)
            ):
                continue
            new_value = self.new_tensor_value(value.const_value.numpy())
            if new_value is not None:
                value.const_value = ir.tensor(new_value, name=value.name)
                modified += 1

        if modified > 0:
            logger.debug("Replaced %d values below threshold with replacement.", modified)
        return model

    def valid_consumers(self, value: ir.Value) -> bool:
        """Check that every consumer of *value* is an allowed op.

        This avoids materializing large tensors that are not attention masks.
        """
        return all(use.node.op_type in self.ALLOWED_CONSUMER_OPS for use in value.uses())

    def new_tensor_value(self, tensor_value: np.ndarray) -> np.ndarray | None:
        """Replace values below the threshold with the replacement value."""
        if np.any(tensor_value < self.threshold):
            tensor_value_new = tensor_value.copy()
            tensor_value_new[tensor_value_new < self.threshold] = self.replacement
            return tensor_value_new
        return None


class RemoveGidxFromMatMulNBits(Surgeon):
    """Drop sorted ``g_idx`` inputs from nodes (e.g. ``MatMulNBits``).

    A ``g_idx`` that is monotonically non-decreasing is an identity channel
    permutation, so it can be removed and the node falls back to the default
    channel order. The now-unused ``g_idx`` initializer is pruned.
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        graph = model.graph
        removed = 0
        for node in list(graph):
            drop = None
            for inp in node.inputs:
                if (
                    inp is not None
                    and inp.name
                    and inp.name.endswith("g_idx")
                    and inp.const_value is not None
                    and np.all(np.diff(inp.const_value.numpy().ravel()) >= 0)
                ):
                    drop = inp
                    break
            if drop is None:
                continue

            # Rebuild the node's inputs without the g_idx entry (shifting later inputs left).
            new_inputs = [inp for inp in node.inputs if inp is not drop]
            for idx, value in enumerate(new_inputs):
                node.replace_input_with(idx, value)
            node.resize_inputs(len(new_inputs))

            if drop.name in graph.initializers and next(iter(drop.uses()), None) is None:
                del graph.initializers[drop.name]
            removed += 1

        if removed:
            logger.debug("Removed g_idx from %d nodes", removed)
        return model


class RemoveQDQ(ProtoSurgeon):
    """Remove QuantizeLinear and DequantizeLinear node pairs from the graph.

    Finds Q->DQ patterns and removes them, directly connecting their inputs/outputs.
    Optionally keeps Clip nodes after graph inputs for value range constraints.
    """

    def __init__(self, keep_clip_after_inputs: bool = False):
        self.keep_clip_after_inputs = keep_clip_after_inputs

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_remove_qdq

        transform_remove_qdq(model, keep_clip_after_inputs=self.keep_clip_after_inputs)
        return model


class MatMulToTransposeConvTranspose(ProtoSurgeon):
    """Replace 2D Gemm/MatMul with Transpose and 1x1 Conv.

    Modification requirement:
    When C==1, convert it to 1x1 Conv using TRANSPOSE + CONV + TRANSPOSE sequence
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_matmul_to_transpose_conv_transpose

        transform_matmul_to_transpose_conv_transpose(model)
        return model


class RemoveIntermediarySqueezeAndUnsqueeze(ProtoSurgeon):
    """Remove all Unsqueeze and Squeeze operations that aren't directly connected to model inputs.

    This optimization removes unnecessary dimension expansion operations in the middle of the graph.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_remove_intermediary_squeeze_and_unsqueeze

        transform_remove_intermediary_squeeze_and_unsqueeze(model)
        return model


class QDQToClip(ProtoSurgeon):
    """Replace QuantizeLinear-DequantizeLinear pairs with Clip operations.

    Converts Q->DQ patterns to Clip nodes with computed min/max values based on
    quantization scale and zero point, maintaining the same value constraints.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_qdq_to_clip

        transform_qdq_to_clip(model)
        return model


class RemoveDeqLin(ProtoSurgeon):
    """Remove DequantizeLinear nodes that operate on constant initializers.

    Dequantizes constant initializers at compile time and replaces DequantizeLinear
    nodes with the pre-computed float values, reducing runtime operations.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_remove_deqlin

        transform_remove_deqlin(model)
        return model


class Non4DModelInputs(ProtoSurgeon):
    """Add Unsqueeze node to model inputs if input is 2D or 3D. Special case is if there's already an Unsqueeze node.

    Ensures all model inputs are 4D by adding Unsqueeze operations:
    - 2D inputs: adds dimensions at positions [0, -1]
    - 3D inputs: adds dimension at position [1]
    Updates existing Unsqueeze nodes if present to maintain 4D output.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_model_inputs

        transform_non4d_model_inputs(model)
        return model


class Non4DModelOutputs(ProtoSurgeon):
    """Add Squeeze to non 4D model outputs.

    Ensures model outputs match expected dimensions by adding Squeeze operations:
    - For 2D outputs: squeeze dimensions [0, 3] or [0, 1]
    - For 3D outputs: squeeze dimension [2]
    Handles special case of Squeeze->Clip->Output pattern.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_model_outputs

        transform_non4d_model_outputs(model)
        return model


class StandaloneReduceSum(ProtoSurgeon):
    """Set keepdims=1 in the ReduceSum attributes, and increment the axes from [1] to [2].

    Modifies standalone ReduceSum operations (not already transformed) to:
    - Set keepdims=1 to preserve dimensions
    - Change reduction axis from [1] to [2] for DLA compatibility
    - Skip if axes is already [-1] (reduce last dimension)

    TODO: use -axes instead of fixed value
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_standalone_reducesum

        transform_standalone_reducesum(model)
        return model


class Gather(ProtoSurgeon):
    """Change Gather indices from scalar to vector, may need to update axis.

    Transforms Gather operations for DLA compatibility:
    - Converts scalar indices to 1D vector format
    - Updates axis attribute from 1 to 2 when needed
    - Ensures indices are always in array format
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_gather

        transform_gather(model)
        return model


class GatherElements(ProtoSurgeon):
    """Change Gather indices from scalar to vector, may need to update axis.

    Transforms GatherElements operations for DLA compatibility:
    - Converts scalar indices to 1D vector format
    - Reshapes 3D indices to 4D by adding dimension at front
    - Ensures indices match expected tensor format
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_gatherelements

        transform_gatherelements(model)
        return model


class Non4DInitializers(ProtoSurgeon):
    """Expand non-4D initializers to 4D format for DLA compatibility.

    Transforms:
    - 1D [K] → [1x1x1xK] for Div/Sub/Mul operations
    - 2D [CxK] → [KxCx1x1] (transpose and reshape) for most operations
    - 2D [K,C] → [1,1,K,C] for Gemm operations
    - 3D → 4D by adding dimension at front

    Skips MatMul inputs and only expands initializers used by specific operations.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_initializers

        transform_non4d_initializers(model)
        return model


class RemoveAllTensorValueShapes(ProtoSurgeon):
    """Remove all tensor shape information from value_info.

    Clears shape fields from all value_info entries in the graph, useful for
    models where shape inference is not needed or causes issues.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_remove_all_tensor_value_shapes

        transform_remove_all_tensor_value_shapes(model)
        return model


class Non4DReshape(ProtoSurgeon):
    """Convert 3D Reshape operations to 4D for DLA compatibility.

    Updates Reshape nodes with 3D target shapes to 4D by inserting 1 at the
    appropriate position (e.g., [-1, 512, 768] -> [1, -1, 512, 768]).
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_reshape

        transform_non4d_reshape(model)
        return model


class Non4DExpand(ProtoSurgeon):
    """Convert 3D Expand operations to 4D for DLA compatibility.

    Updates Expand nodes with 3D shapes to 4D by inserting 1 at dimension 0
    (e.g., [2, 3, 4] -> [1, 2, 3, 4]).
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_expand

        transform_non4d_expand(model)
        return model


class Non4DTranspose(ProtoSurgeon):
    """Update Transpose permutation attributes for non-4D tensors.

    Adjusts perm attribute of Transpose nodes to handle 4D tensors:
    - 2D: [T0, T1] -> [0, 1, T0 + 2, T1 + 2]
    - 3D: [T0, T1, T2] -> [0, T0 + 1, T1 + 1, T2 + 1]

    Ensures transpose operations work correctly when tensors are expanded to 4D.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_transpose

        transform_non4d_transpose(model)
        return model


class Non4DSlice(ProtoSurgeon):
    """Transform Slice axes of non4D tensors.

    Updates Slice operations to work with 4D tensors:
    - Changes axes from specific dimensions to [-1] (last dimension)
    - Only processes nodes not already transformed
    - Ensures slicing operations remain valid after tensor expansion
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_slice

        transform_non4d_slice(model)
        return model


class Non4DLpNorm(ProtoSurgeon):
    """Transform LpNormalization axes of non4D tensors.

    Updates LpNormalization operations for 4D tensor compatibility:
    - Changes axis attribute to -1 (last dimension)
    - Ensures normalization occurs along the correct dimension
    after tensor expansion to 4D
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_non4d_lpnorm

        transform_non4d_lpnorm(model)
        return model


class Flatten(ProtoSurgeon):
    """Flatten to Reshape.

    Replaces Flatten operations with Reshape operations using shape [1, 1, 1, -1].
    This maintains compatibility with DLA which may not support Flatten directly,
    while preserving the flattening behavior.
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_flatten

        transform_flatten(model)
        return model


class AddIntermediateTensorsToOutputs(ProtoSurgeon):
    """Debug function to add intermediate tensors to outputs.

    Exposes intermediate tensor values as model outputs for debugging:
    - Can specify specific tensors to add via intermediate_tensor_to_add list
    - If not specified, adds all intermediate tensors from node outputs
    - Useful for inspecting values at different stages of the graph
    """

    def __init__(self, intermediate_tensor_to_add: list = None):
        self.intermediate_tensor_to_add = intermediate_tensor_to_add

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_add_intermediate_tensors_to_outputs

        transform_add_intermediate_tensors_to_outputs(model, intermediate_tensor_to_add=self.intermediate_tensor_to_add)
        return model


class ReshapeReduceSum(ProtoSurgeon):
    """Transform Reshape-ReduceSum pattern to parallel Slice-ReduceSum-Concat for DLA.

    Splits a Reshape-ReduceSum operation into parallel paths to improve DLA performance:
    - Replaces single Reshape-ReduceSum with two parallel Slice operations
    - Each slice processes part of the data with ReduceSum
    - Results are concatenated to produce the same output
    - Enables better parallelization on DLA hardware
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_reshape_reducesum

        transform_reshape_reducesum(model)
        return model


class ReshapeClipReduceSum(ProtoSurgeon):
    """Transform Reshape-Clip-ReduceSum pattern to parallel paths for DLA optimization.

    Similar to ReshapeReduceSum but includes Clip operation:
    - Splits Reshape-Clip-ReduceSum into two parallel processing paths
    - Each path: Slice -> Clip -> ReduceSum
    - Maintains numerical equivalence while improving DLA parallelization
    - Useful for quantized models where Clip enforces value ranges
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_reshape_clip_reducesum

        transform_reshape_clip_reducesum(model)
        return model


class ReduceMax(ProtoSurgeon):
    """Add Reshape after ReduceMax operations for DLA compatibility.

    Modifies ReduceMax operations to ensure output shape compatibility:
    - Adds a Reshape node after ReduceMax with shape [1,1,1,3600]
    - Updates axes to [3] and keepdims to 1
    - Ensures ReduceMax output has the expected 4D shape for DLA
    - Hardcoded output shape may need adjustment for different models
    """

    def __call__(self, model: ModelProto):
        from olive.passes.onnx.dla_transforms import transform_reducemax

        transform_reducemax(model)
        return model


class TieWordEmbeddings(ProtoSurgeon):
    """Tie word embeddings.

    Only supports when the embeddings are not quantized or when both are quantized
    using GatherBlockQuantized and MatMulNBits
    """

    def __call__(self, model: onnx.ModelProto):
        dag = OnnxDAG(model)

        # support both "input_ids" and "input_embeds"/"inputs_embeds" as input names
        input_name = None
        for candidate in ("input_ids", "input_embeds", "inputs_embeds"):
            if candidate in dag.ios and dag.is_input(candidate):
                input_name = candidate
                break

        if input_name is None or "logits" not in dag.ios or not dag.is_output("logits"):
            return dag.model

        embed_name, embed_op_type = self.get_name_op_type(
            dag, dag.get_consumers(input_name), ["Gather", "GatherBlockQuantized"], 0
        )
        if embed_name is None:
            return dag.model

        lm_head_name, lm_head_op_type = self.get_name_op_type(
            dag, [dag.get_producer("logits")], ["MatMul", "MatMulNBits"], 1
        )
        if lm_head_name is None:
            return dag.model

        # skip if one is quantized and the other is not
        if (embed_op_type, lm_head_op_type) not in [
            ("Gather", "MatMul"),
            ("GatherBlockQuantized", "MatMulNBits"),
        ]:
            return dag.model

        if embed_op_type == "Gather" and lm_head_op_type == "MatMul":
            return self.handle_unquantized(dag, embed_name, lm_head_name)
        return self.handle_quantized(dag, embed_name, lm_head_name)

    def get_name_op_type(
        self, dag: OnnxDAG, candidates: list[str], supported_ops: list[str], input_idx: int
    ) -> tuple[str | None, str | None]:
        for candidate in candidates:
            op_type = dag.get_node_op_type(candidate)
            if op_type not in supported_ops:
                continue
            if not dag.is_initializer(dag.get_node_inputs(candidate)[input_idx]):
                continue
            return candidate, op_type
        return None, None

    def handle_unquantized(self, dag: OnnxDAG, embed_name: str, lm_head_name: str):
        embed_weight_name = dag.get_node_inputs(embed_name)[0]
        lm_head_weight_name = dag.get_node_inputs(lm_head_name)[1]

        if not self.equal_weights(
            dag,
            embed_weight_name,
            lm_head_weight_name,
            transpose=True,
        ):
            logger.debug("Weights are not the same, cannot tie them")
            return dag.model

        logger.debug("Tying weights")
        out_feat, in_feat = dag.get_io_shape(embed_weight_name)

        graph_idx = dag.get_graph_idx(lm_head_name)

        lm_head_output_name = dag.get_node_outputs(lm_head_name)[0]
        lm_head_input_name = dag.get_node_inputs(lm_head_name)[0]

        # add a reshape before the lm head to make the input 2D
        prereshape_name = self.create_new_name(lm_head_name, "MatMul", "Reshape_pre")
        prereshape_output_name = self.add_reshape_node(
            dag, graph_idx, prereshape_name, lm_head_input_name, [-1, in_feat]
        )

        # add gemm node to replace matmul
        gemm_name = lm_head_name.replace("MatMul", "Gemm")
        gemm_output_name = f"{gemm_name}_output"
        dag.add_node(
            onnx.helper.make_node(
                "Gemm",
                [prereshape_output_name, embed_weight_name],
                [gemm_output_name],
                name=gemm_name,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1,
            ),
            graph_idx,
        )

        # need to get the original input shape to reshape back
        input_shape_name = lm_head_name.replace("MatMul", "Shape")
        input_shape_output_name = f"{input_shape_name}_output"
        dag.add_node(
            onnx.helper.make_node(
                "Shape",
                [lm_head_input_name],
                [input_shape_output_name],
                name=input_shape_name,
            ),
            graph_idx,
        )

        # get all but the last dimension from the input shape
        slice_name = self.create_new_name(lm_head_name, "MatMul", "Slice")
        slice_output_name = f"{slice_name}_output"
        slice_input_names = [input_shape_output_name]
        for key, value in {
            "start": np.array([0], dtype=np.int64),
            "ends": np.array([-1], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
            "steps": np.array([1], dtype=np.int64),
        }.items():
            initializer_name = f"{slice_name}_{key}"
            dag.add_initializer(onnx.numpy_helper.from_array(value, initializer_name), graph_idx)
            slice_input_names.append(initializer_name)
        dag.add_node(
            onnx.helper.make_node(
                "Slice",
                slice_input_names,
                [slice_output_name],
                name=slice_name,
            ),
            graph_idx,
        )

        # output shape is concat of sliced shape and out_feat
        concat_name = self.create_new_name(lm_head_name, "MatMul", "Concat")
        concat_output_name = f"{concat_name}_output"
        out_feat_init_name = f"{concat_name}_out_feat"
        dag.add_initializer(
            onnx.numpy_helper.from_array(np.array([out_feat], dtype=np.int64), out_feat_init_name), graph_idx
        )
        dag.add_node(
            onnx.helper.make_node(
                "Concat",
                [slice_output_name, out_feat_init_name],
                [concat_output_name],
                name=concat_name,
                axis=0,
            ),
            graph_idx,
        )

        # add reshape after gemm to restore original shape
        post_reshape_name = self.create_new_name(lm_head_name, "MatMul", "Reshape_post")
        post_reshape_output_name = self.add_reshape_node(
            dag,
            graph_idx,
            post_reshape_name,
            gemm_output_name,
            concat_output_name,
        )
        post_reshape_output_vi = onnx.ValueInfoProto()
        post_reshape_output_vi.CopyFrom(dag.get_value_info_proto(lm_head_output_name))
        post_reshape_output_vi.name = post_reshape_output_name
        dag.add_value_info(post_reshape_output_vi, graph_idx)

        # redirect consumers of lm head output to post reshape output
        for consumer in dag.get_consumers(lm_head_name):
            dag.replace_node_input(consumer, lm_head_output_name, post_reshape_output_name)

        # remove logits output and lm head node
        dag.remove_output(lm_head_output_name)
        dag.remove_node(lm_head_name)
        # rename post reshape output to logits and make it output again
        dag.rename_node_output(post_reshape_name, post_reshape_output_name, lm_head_output_name)
        dag.make_output(lm_head_output_name)

        dag.update()
        return dag.model

    def handle_quantized(self, dag: OnnxDAG, embed_name: str, lm_head_name: str):
        embed_inputs = dag.get_node_inputs(embed_name)
        lm_head_inputs = dag.get_node_inputs(lm_head_name)
        if len(embed_inputs) != len(lm_head_inputs):
            logger.debug("Different number of inputs. Cannot tie embeddings.")
            return dag.model

        graph_idx = dag.get_graph_idx(lm_head_name)

        embed_inputs = embed_inputs[0:1] + embed_inputs[2:]
        lm_head_inputs = lm_head_inputs[1:]

        for embed_init, lm_head_init in zip(embed_inputs, lm_head_inputs):
            # won't support old gatherblockquantized which uses uint4 data, it has different packing
            if not self.equal_weights(dag, embed_init, lm_head_init):
                logger.debug("Initializer %s and %s are not equal. Cannot tie embeddings.", embed_init, lm_head_init)
                return dag.model

        logger.debug("Tying quantized weights")

        # point both to the same scales and zero points, use the embedding ones since it has 2D shapes
        # some matmulnbits quantizers still use old 1D scales/zero points
        for embed_init, lm_head_init in zip(embed_inputs[1:], lm_head_inputs[1:]):
            if embed_init == lm_head_init:
                continue
            dag.replace_node_input(lm_head_name, lm_head_init, embed_init)

        # need a reshape for the qweight, MatNBits expects 3D weights but GatherBlockQuantized uses 2D weights
        # doesn't matter which one since ORT constant folds the reshape and duplicates the initializer during session creation
        reshape_output = self.add_reshape_node(
            dag,
            graph_idx,
            self.create_new_name(lm_head_name, "MatMulNBits", "Reshape_tied"),
            embed_inputs[0],
            dag.get_io_shape(lm_head_inputs[0]),
            dag.get_io_elem_type(lm_head_inputs[0]),
        )
        dag.replace_node_input(lm_head_name, lm_head_inputs[0], reshape_output)

        dag.update()
        return dag.model

    def equal_weights(self, dag: OnnxDAG, init0: str, init1: str, transpose: bool = False) -> bool:
        shape0, shape1 = dag.get_io_shape(init0), dag.get_io_shape(init1)
        if np.prod(shape0) != np.prod(shape1):
            # this will fail if GatherBlockQuantized uses uint4 packing, our quantizer doesn't use that
            # only case is the onnx default mnb quantizer but it uses different algos for gather vs matmul
            # so weight don't match anyway
            return False

        arr0 = dag.get_initializer_np_array(init0)
        arr1 = dag.get_initializer_np_array(init1)
        if transpose:
            arr0 = arr0.T
        return np.array_equal(arr0.ravel(), arr1.ravel())


class QuantizeEmbeddingInt8(ProtoSurgeon):
    """Quantize FP16 embedding to INT8 using GatherBlockQuantized.

    Replaces the Gather op for embed_tokens with a GatherBlockQuantized op
    that uses per-block INT8 quantization (block_size=32).
    """

    def __call__(self, model: onnx.ModelProto):
        from onnx import numpy_helper

        dag = OnnxDAG(model)

        # Find embedding Gather node
        gather_name = self.find_node(dag, "Gather", "embed_tokens")
        if gather_name is None:
            logger.warning("No embed_tokens Gather node found, skipping QuantizeEmbeddingInt8")
            return model

        embed_weight_name = dag.get_node_inputs(gather_name)[0]
        if not dag.is_initializer(embed_weight_name):
            logger.warning("Embedding weight initializer not found, skipping QuantizeEmbeddingInt8")
            return model

        embed_init = dag.get_initializer_proto(embed_weight_name)

        # Check if already quantized
        if embed_init.data_type not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT):
            logger.info("Embedding is not FP16/FP32, skipping QuantizeEmbeddingInt8")
            return model

        embed = dag.get_initializer_np_array(embed_weight_name).astype(np.float32)
        vocab_size, hidden_size = embed.shape
        block_size = 32

        if hidden_size % block_size != 0:
            logger.warning("hidden_size %d not divisible by block_size %d, skipping", hidden_size, block_size)
            return model

        num_blocks = hidden_size // block_size

        # Preserve the model's float dtype for scales so downstream ops (LayerNorm, MatMul, ...)
        # receive the dtype they expect. FP16 model -> FP16 scales; FP32 model -> FP32 scales.
        scales_dtype = np.float16 if embed_init.data_type == onnx.TensorProto.FLOAT16 else np.float32

        logger.info(
            "Quantizing embedding %s (%dx%d) from %s to INT8 (block_size=%d)",
            embed_weight_name,
            vocab_size,
            hidden_size,
            "FP16" if scales_dtype == np.float16 else "FP32",
            block_size,
        )

        # Per-block INT8 quantization (asymmetric with zero_point=128 for GatherBlockQuantized)
        blocked = embed.reshape(vocab_size, num_blocks, block_size)
        scales = (np.abs(blocked).max(axis=2) / 127.0).astype(scales_dtype)
        scales_f32 = scales.astype(np.float32)
        # Avoid division by zero
        scales_f32 = np.where(scales_f32 == 0, 1.0, scales_f32)
        q = np.clip(np.round(blocked / scales_f32[:, :, None]), -128, 127).astype(np.int8)
        # GatherBlockQuantized expects unsigned uint8 with zero_point offset
        q_uint8 = (q.astype(np.int16) + 128).astype(np.uint8)
        q_flat = q_uint8.reshape(vocab_size, hidden_size)
        # Zero point tensor: 128 for all blocks (symmetric around 128)
        zero_points = np.full((vocab_size, num_blocks), 128, dtype=np.uint8)

        old_size_mb = embed.nbytes / (1024 * 1024)
        new_size_mb = (q_flat.nbytes + scales.nbytes + zero_points.nbytes) / (1024 * 1024)
        logger.info(
            "Embedding: %.0f MB -> %.0f MB (saved %.0f MB)", old_size_mb, new_size_mb, old_size_mb - new_size_mb
        )

        graph_idx = dag.get_graph_idx(gather_name)

        # Create new initializers
        qweight_name = embed_weight_name + "_Q8"
        scales_name = embed_weight_name + "_scales"
        zp_name = embed_weight_name + "_zp"
        dag.add_initializer(numpy_helper.from_array(q_flat, name=qweight_name), graph_idx)
        dag.add_initializer(numpy_helper.from_array(scales, name=scales_name), graph_idx)
        dag.add_initializer(numpy_helper.from_array(zero_points, name=zp_name), graph_idx)

        # Ensure com.microsoft opset is declared
        dag.set_opset_import("com.microsoft", 1)

        # Create GatherBlockQuantized node
        gather_inputs = dag.get_node_inputs(gather_name)
        gather_output = dag.get_node_outputs(gather_name)[0]
        gbq_output = gather_output + "_gbq"
        gbq_name = gather_name.replace("Gather", "GatherBlockQuantized")
        gbq_node = onnx.helper.make_node(
            "GatherBlockQuantized",
            inputs=[qweight_name, gather_inputs[1], scales_name, zp_name],
            outputs=[gbq_output],
            name=gbq_name,
            domain="com.microsoft",
            bits=8,
            block_size=block_size,
            gather_axis=0,
            quantize_axis=1,
        )
        dag.add_node(gbq_node, graph_idx)

        # Rewire consumers from old Gather output to new GBQ output and remove old node
        for consumer in dag.get_consumers(gather_output):
            dag.replace_node_input(consumer, gather_output, gbq_output)
        dag.remove_node(gather_name)
        # Old FP16 embedding weight is auto-cleaned by update() since no consumers remain

        logger.info("Replaced Gather with GatherBlockQuantized (INT8)")
        dag.update()
        return dag.model


class ShareEmbeddingLmHead(ProtoSurgeon):
    """Share INT8 embedding weight with lm_head by converting lm_head to INT8 MatMulNBits.

    Must be applied AFTER QuantizeEmbeddingInt8. Replaces the lm_head's INT4
    MatMulNBits with an INT8 MatMulNBits that references the same quantized
    weight as the embedding's GatherBlockQuantized, eliminating duplicate storage.
    """

    def __call__(self, model: onnx.ModelProto):
        from onnx import numpy_helper

        dag = OnnxDAG(model)

        # Find embedding GatherBlockQuantized
        gbq_name = self.find_node(dag, "GatherBlockQuantized", "embed_tokens")
        if gbq_name is None:
            logger.warning("No embed_tokens GatherBlockQuantized node found, skipping ShareEmbeddingLmHead")
            return model

        attrs = dag.get_node_attributes(gbq_name)
        gbq_bits = attrs.get("bits", 8)
        gbq_block_size = attrs.get("block_size", 32)

        if gbq_bits != 8:
            logger.warning("Embedding is not INT8, cannot share with lm_head")
            return model

        # Get embedding weight, scales, zero_points names
        gbq_inputs = dag.get_node_inputs(gbq_name)
        embed_weight_name = gbq_inputs[0]
        embed_scales_name = gbq_inputs[2]
        embed_zp_name = gbq_inputs[3] if len(gbq_inputs) > 3 else None

        # Get embedding weight shape to determine K and N
        if not dag.is_initializer(embed_weight_name):
            logger.warning("Could not find embedding weight initializer")
            return model

        embed_weight = dag.get_initializer_np_array(embed_weight_name)

        vocab_size, hidden_size = embed_weight.shape  # [V, H] for INT8
        num_blocks = hidden_size // gbq_block_size

        # Find lm_head MatMulNBits node
        lm_head_name = self.find_node(dag, "MatMulNBits", "lm_head")
        if lm_head_name is None:
            logger.warning("No lm_head MatMulNBits found")
            return model

        lm_head_inputs = dag.get_node_inputs(lm_head_name)

        # Check if already shared (idempotency): lm_head weight input references embedding weight
        if embed_weight_name in lm_head_inputs[1] or lm_head_inputs[2] == embed_scales_name:
            logger.info("lm_head already shares weights with embedding, skipping ShareEmbeddingLmHead")
            return model

        # Get old lm_head attributes
        old_attrs = dag.get_node_attributes(lm_head_name)

        logger.info(
            "Sharing embedding with lm_head: lm_head INT%d (%dx%d, bs=%d) -> INT8 (shared with embedding)",
            old_attrs.get("bits", 0),
            old_attrs.get("N", 0),
            old_attrs.get("K", 0),
            old_attrs.get("block_size", 0),
        )

        graph_idx = dag.get_graph_idx(lm_head_name)

        # MatMulNBits needs [N, K_blocks, block_size] but GBQ weight is [V, H].
        # Add a Reshape node to convert, referencing the SAME embedding weight.
        reshape_shape_name = "lm_head.MatMulNBits.reshape_shape"
        reshape_shape = np.array([vocab_size, num_blocks, gbq_block_size], dtype=np.int64)
        dag.add_initializer(numpy_helper.from_array(reshape_shape, name=reshape_shape_name), graph_idx)

        reshape_output = "lm_head.MatMulNBits.reshaped_weight"
        reshape_node = onnx.helper.make_node(
            "Reshape",
            inputs=[embed_weight_name, reshape_shape_name],
            outputs=[reshape_output],
            name="lm_head/Reshape_shared_weight",
        )
        dag.add_node(reshape_node, graph_idx)

        # Scales and zp: reuse embedding's directly
        inputs = [lm_head_inputs[0], reshape_output, embed_scales_name]
        if embed_zp_name:
            inputs.append(embed_zp_name)

        # Ensure com.microsoft opset is declared
        dag.set_opset_import("com.microsoft", 1)

        # Create new INT8 MatMulNBits node
        lm_head_output = dag.get_node_outputs(lm_head_name)[0]
        new_lm_head_output = lm_head_output + "_shared"
        new_lm_head_name = lm_head_name + "_shared"
        lm_head_proto = dag.get_node_proto(lm_head_name)
        new_lm_head = onnx.helper.make_node(
            "MatMulNBits",
            inputs=inputs,
            outputs=[new_lm_head_output],
            name=new_lm_head_name,
            domain="com.microsoft",
            bits=8,
            block_size=gbq_block_size,
            K=hidden_size,
            N=vocab_size,
        )
        # Copy accuracy_level if present
        for attr in lm_head_proto.attribute:
            if attr.name == "accuracy_level":
                new_lm_head.attribute.append(attr)

        dag.add_node(new_lm_head, graph_idx)

        # Copy value info from old output to new output (needed for graph output serialization)
        old_vi = dag.get_value_info_proto(lm_head_output)
        if old_vi is not None:
            new_vi = onnx.helper.make_tensor_value_info(new_lm_head_output, old_vi.type.tensor_type.elem_type, [])
            new_vi.CopyFrom(old_vi)
            new_vi.name = new_lm_head_output
            dag.add_value_info(new_vi, graph_idx)

        # Rewire consumers and remove old node
        for consumer in dag.get_consumers(lm_head_output):
            dag.replace_node_input(consumer, lm_head_output, new_lm_head_output)
        if dag.is_output(lm_head_output):
            dag.remove_output(lm_head_output)
            dag.remove_node(lm_head_name)
            dag.rename_node_output(new_lm_head_name, new_lm_head_output, lm_head_output)
            dag.make_output(lm_head_output)
        else:
            dag.remove_node(lm_head_name)
        # Old lm_head initializers are auto-cleaned by update() since no consumers remain

        logger.info("lm_head now uses INT8 MatMulNBits (shared quantization with embedding)")
        dag.update()
        return dag.model


class ReciprocalMulToDiv(RewriteRuleSurgeon):
    """Replace Reciprocal(x) * a  with  Div(a, x).

    Before:
        [x] --> Reciprocal --> Mul --> [out]
                               ^
                               |
                              [a]

    After:
        [a] --> Div --> [out]
                 ^
                 |
                [x]

    Why this is needed:
        PyTorch's ``torch.rsqrt()`` (used by Qwen2.5-VL's ``Qwen2RMSNorm``) decomposes to
        ``Sqrt -> Reciprocal -> Mul`` in ONNX.  ORT's ``SimplifiedLayerNormFusion`` only
        matches the pattern ``Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul`` — it does
        **not** recognize the ``Reciprocal -> Mul`` variant (confirmed on ORT main as of
        2025-06).  This pass canonicalizes the graph so that the fusion fires, replacing
        decomposed RMSNorm with a single ``SimplifiedLayerNormalization`` op.

    When to use:
        Run **before** ``OrtTransformersOptimization`` on models whose normalization layers
        export ``rsqrt`` as ``Reciprocal`` (e.g. HuggingFace models using ``torch.rsqrt``).
    """

    def rules(self) -> pattern.RewriteRuleSet:
        def _pattern(op, x, a):
            # Match a * Reciprocal(x); commute=True also matches Reciprocal(x) * a.
            return op.Mul(a, op.Reciprocal(x))

        def _replacement(op, x, a):
            return op.Div(a, x)

        return pattern.RewriteRuleSet([pattern.RewriteRule(_pattern, _replacement)], commute=True)


class DeduplicateSubgraphInitializers(ProtoSurgeon):
    """Remove duplicate initializers in Loop / If / Scan subgraphs.

    Why this is needed:
        ORT's graph optimizer (constant folding, shape inference, etc.) may copy
        initializers into subgraphs that already contain them, creating entries with
        identical names.  ORT's ``ConstantSharing`` pass explicitly skips subgraph
        usage (``constant_sharing.cc``: "If usage is from subgraph, skip it now"),
        so these duplicates are never cleaned up.  Duplicate initializers violate
        the ONNX spec's unique-name requirement and can cause validation failures
        or silent data corruption.

    What it does:
        For every ``Loop`` / ``If`` / ``Scan`` subgraph, keeps the first initializer
        with a given name and removes all subsequent duplicates.

    When to use:
        Run **after** ``OrtTransformersOptimization`` (which introduces the duplicates)
        and **before** any pass that serializes or validates the model.
    """

    def __call__(self, model: ModelProto):
        removed = 0
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.g and attr.g.initializer:
                    seen = set()
                    to_remove = []
                    for init in attr.g.initializer:
                        if init.name in seen:
                            to_remove.append(init)
                        else:
                            seen.add(init.name)
                    for init in to_remove:
                        attr.g.initializer.remove(init)
                        removed += 1
        if removed > 0:
            logger.debug("Removed %d duplicate subgraph initializers", removed)
        return model


class DeduplicateNodes(ProtoSurgeon):
    """Remove nodes whose output tensors are already produced by an earlier node.

    Before (invalid — two nodes define the same tensor ``/Cast_output_0``):
        NodeA  -->  Cast  --> /Cast_output_0
        NodeB  -->  Cast  --> /Cast_output_0   (duplicate, removed)

    After:
        NodeA  -->  Cast  --> /Cast_output_0

    Why this is needed:
        ORT's ``convert_float_to_float16`` (``float16.py``) may insert identical
        ``Cast`` nodes in parallel branches that each declare the same output tensor
        name.  The ONNX spec requires every tensor to have a unique producer; loading
        a model with duplicate producers causes ``onnxruntime.InferenceSession`` to
        fail with a duplicate-definition error.

    What it does:
        Scans nodes in graph order and records each output tensor name.  If a later
        node produces a tensor name that was already seen, the entire node is removed.

    When to use:
        Run **after** ``OnnxFloatToFloat16`` as a cleanup step.
    """

    def __call__(self, model: ModelProto):
        output_seen: set[str] = set()
        indices_to_remove: list[int] = []
        for i, node in enumerate(model.graph.node):
            dup = False
            for o in node.output:
                if o and o in output_seen:
                    dup = True
                    break
                if o:
                    output_seen.add(o)
            if dup:
                indices_to_remove.append(i)
        for i in reversed(indices_to_remove):
            del model.graph.node[i]
        if indices_to_remove:
            logger.debug("Removed %d duplicate nodes", len(indices_to_remove))
        return model


class PackedAttentionToLoopMHA(Surgeon):
    """Replace custom::PackedAttention with a loop calling com.microsoft::MultiHeadAttention.

    This surgery expands the custom PackedAttention operation into a loop that processes
    each sequence segment separately using MultiHeadAttention.

    Input shapes:
        - query_states, key_states, value_states: [B, num_heads, seq_len, head_dim]
        - cu_seqlens: [num_segments + 1] cumulative sequence lengths

    Output shape:
        - attn_output: [B, seq_len, num_heads, head_dim]
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        # Get the opset version from the model
        opset_version = model.opset_imports.get("", 20)

        custom = onnxscript.values.Opset(OpType.Custom, 1)
        op = onnxscript.values.Opset("", opset_version)
        msft_op = onnxscript.values.Opset(MSFT_DOMAIN, 1)

        @onnxscript.script(opset=custom)
        def PackedAttention(  # noqa: N802
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            scale: float,
            num_heads: int,
        ):
            # Shapes of input Q/K/V: [B, num_heads, seq_len, head_dim]

            # Convert Q/K/V to shape [B, seq_len, num_heads*head_dim]
            to_3d_shape = op.Constant(value_ints=[0, 0, -1])
            query_transposed = op.Transpose(query_states, perm=[0, 2, 1, 3])
            output_shape = op.Shape(query_transposed)
            query_3d = op.Reshape(query_transposed, to_3d_shape)
            value_3d = op.Reshape(op.Transpose(value_states, perm=[0, 2, 1, 3]), to_3d_shape)
            key_3d = op.Reshape(op.Transpose(key_states, perm=[0, 2, 1, 3]), to_3d_shape)

            num_patches = op.Size(cu_seqlens) - 1
            seq_axis = op.Constant(value_ints=[1])
            seq_axis_int32 = op.Cast(seq_axis, to=onnx.TensorProto.INT32)
            attn_output = op.Slice(value_3d, [0], [0], seq_axis)  # Initialize empty output
            for i in range(num_patches):
                i_1d = op.Reshape(i, [1])
                i_plus_1_1d = i_1d + 1
                start = op.Gather(cu_seqlens, i_1d, axis=0)
                end = op.Gather(cu_seqlens, i_plus_1_1d, axis=0)

                query_i = op.Slice(query_3d, start, end, seq_axis_int32)
                key_i = op.Slice(key_3d, start, end, seq_axis_int32)
                value_i = op.Slice(value_3d, start, end, seq_axis_int32)

                mha_output = msft_op.MultiHeadAttention(
                    query_i,
                    key_i,
                    value_i,
                    num_heads=num_heads,
                    scale=scale,
                )
                attn_output = op.Concat(attn_output, mha_output, axis=1)
            return op.Reshape(attn_output, output_shape)  # [B, seq_len, num_heads, head_dim]

        # Update the functions into the model
        irfunctions: list[ir.Function] = [ir.from_proto(PackedAttention.to_function_proto())]
        model_functions = model.functions

        if len(model_functions) != 0:
            raise ValueError("Input model cannot have model-local functions.")
        for func in irfunctions:
            model_functions[func.identifier()] = func

        InlinePass()(model)
        RemoveUnusedOpsetsPass()(model)
        return model


class PackedAttentionToPackedMHA(Surgeon):
    """Replace custom::PackedAttention with com.microsoft::PackedMultiHeadAttention.

    This surgery expands the custom PackedAttention operation into a single
    PackedMultiHeadAttention call with computed token offsets.

    Input shapes:
        - query, key, value: [B=1, num_heads, seq_len, head_dim]
        - cu_seqlens: [num_segments + 1] cumulative sequence lengths

    Output shape:
        - attn_output: [B, seq_len, num_heads, head_dim]
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        # Get the opset version from the model
        opset_version = model.opset_imports.get("", 20)

        custom = onnxscript.values.Opset(OpType.Custom, 1)
        op = onnxscript.values.Opset("", opset_version)
        msft_op = onnxscript.values.Opset(MSFT_DOMAIN, 1)

        @onnxscript.script(opset=custom)
        def PackedAttention(query, key, value, cu_seqlens, scale: float, num_heads: int):  # noqa: N802
            # Shapes of input Q/K/V: [B=1, num_heads, seq_len, head_dim]
            num_patches = op.Cast(op.Size(cu_seqlens), to=onnx.TensorProto.INT32) - 1
            # Identify lengths of each patch and max length
            starts = op.Slice(cu_seqlens, [0], [-1], [0])  # [num_patches]
            ends = op.Slice(cu_seqlens, [1], [9223372036854775807], [0])  # [num_patches]
            lengths = ends - starts  # [num_patches]
            max_length = op.ReduceMax(lengths, [0], keepdims=0)  # [1]
            # Create token_offset required by the PackedMultiHeadAttention op
            # First create matrix: [
            #    [0, 1, 2, ..., max_length-1],
            #    [max_length, max_length+1, ..., 2*max_length-1],
            #    ... ]
            rows = op.Range(0, num_patches, 1)  # [num_patches]
            rows_2d = op.Unsqueeze(rows, [1])  # [num_patches, 1]
            cols = op.Range(0, max_length, 1)  # [max_length]
            cols_2d = op.Unsqueeze(cols, [0])  # [1, max_length]
            position_matrix = rows_2d * max_length + cols_2d  # [num_patches, max_length]
            position_matrix_shape = op.Shape(position_matrix)
            # Now find positions of valid tokens and padding tokens
            # Position at column j in row i is valid if j < lengths[i]
            token_mask = cols_2d < op.Unsqueeze(lengths, [1])  # [num_patches, max_length]
            token_mask_1d = op.Reshape(token_mask, [-1])  # [num_patches * max_length]
            # All other positions are padding
            padded_mask_1d = op.Not(token_mask_1d)
            valid_token_positions = op.Compress(position_matrix, token_mask)  # [total_valid_tokens]
            padded_token_positions = op.Compress(position_matrix, padded_mask_1d)  # [total_padded_tokens]
            token_offset_1d = op.Concat(
                valid_token_positions, padded_token_positions, axis=0
            )  # [num_patches * max_length]
            token_offset = op.Reshape(token_offset_1d, position_matrix_shape)  # [num_patches, max_length]

            # Convert query/key/value to shape (seq_len, num_heads* head_dim)
            # squeeze(0) => transpose(0,1) => reshape([0, -1])
            query_3d = op.Transpose(op.Squeeze(query, [0]), perm=[1, 0, 2])
            shape_3d = op.Shape(query_3d)
            query_2d = op.Reshape(query_3d, [0, -1])
            key_2d = op.Reshape(op.Transpose(op.Squeeze(key, [0]), perm=[1, 0, 2]), [0, -1])
            value_2d = op.Reshape(op.Transpose(op.Squeeze(value, [0]), perm=[1, 0, 2]), [0, -1])

            packed_attn_output_2d = msft_op.PackedMultiHeadAttention(
                query_2d,
                key_2d,
                value_2d,
                None,
                token_offset,
                cu_seqlens,
                scale=scale,
                num_heads=num_heads,
            )
            packed_attn_output_3d = op.Reshape(packed_attn_output_2d, shape_3d)
            return op.Unsqueeze(packed_attn_output_3d, [0])  # [B, seq_len, num_heads, head_dim]

        # Update the functions into the model
        irfunctions: list[ir.Function] = [ir.from_proto(PackedAttention.to_function_proto())]
        model_functions = model.functions

        if len(model_functions) != 0:
            raise ValueError("Input model cannot have model-local functions.")
        for func in irfunctions:
            model_functions[func.identifier()] = func

        InlinePass()(model)
        RemoveUnusedOpsetsPass()(model)
        return model


class RenameOutputDims(Surgeon):
    """Rename dynamic dimension names in output shapes.

    This surgery renames the dimension name at a specific index in an output's shape.
    Useful for restoring meaningful dimension names after graph transformations
    that may have changed them.

    Example usage:
        {
            "surgeon": "RenameOutputDims",
            "output_idx": 0,
            "dim_idx": 0,
            "dim_name": "num_logical_patches"
        }
    """

    def __init__(self, output_idx: int, dim_idx: int, dim_name: str):
        super().__init__()
        self.output_idx = output_idx
        self.dim_idx = dim_idx
        self.dim_name = dim_name

    def call_ir(self, model: ir.Model) -> ir.Model:
        outputs = model.graph.outputs
        if self.output_idx >= len(outputs):
            raise ValueError(f"output_idx {self.output_idx} is out of range. Model has {len(outputs)} outputs.")

        output = outputs[self.output_idx]
        if output.shape is None:
            raise ValueError(f"Output at index {self.output_idx} has no shape information.")

        if self.dim_idx >= len(output.shape):
            raise ValueError(f"dim_idx {self.dim_idx} is out of range. Output has {len(output.shape)} dimensions.")

        # Create a new shape with the modified dimension name
        new_dims = list(output.shape)
        new_dims[self.dim_idx] = self.dim_name
        output.shape = ir.Shape(new_dims)
        return model


class RemoveMemcpy(Surgeon):
    """Remove MemcpyToHost and MemcpyFromHost nodes from the graph.

    These nodes are inserted by ORT's ``OrtTransformersOptimization`` when it
    pre-partitions the graph for a GPU execution provider.  They represent
    explicit GPU↔CPU data copies for tensors whose consumers require CPU memory
    (e.g. shape arguments to Reshape, start/end for Slice, trip counts for Loop).

    Removing them is safe because ORT's runtime ``MemcpyTransformer`` will
    re-insert only the truly necessary copies when the session is created.
    The runtime also has a ``GetCpuPreferredNodes`` heuristic that may keep
    entire shape-computation subgraphs on CPU, potentially avoiding some
    copies entirely.

    This surgery processes both the main graph and all Loop/If subgraphs
    recursively.  After removal the graph nodes are topologically re-sorted
    to satisfy the ONNX requirement that every input is produced before use.

    When to use:
        Run **after** ``OrtTransformersOptimization`` to remove pre-baked memcpy
        nodes and let ORT's runtime re-partition optimally.
    """

    def call_ir(self, model: ir.Model) -> ir.Model:
        total = self._remove_from_graph(model.graph)
        if total:
            # Bypassing Memcpy nodes can leave the graph out of topological order.
            TopologicalSortPass()(model)
            logger.debug("Removed %d Memcpy nodes total", total)
        return model

    @classmethod
    def _remove_from_graph(cls, graph: ir.Graph) -> int:
        """Bypass and remove 1-in/1-out MemcpyToHost/MemcpyFromHost nodes, recursively."""
        removed = 0
        for node in list(graph):
            # Recurse into Loop/If/Scan subgraphs first.
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    removed += cls._remove_from_graph(attr.value)
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        removed += cls._remove_from_graph(subgraph)

            if (
                node.op_type not in ("MemcpyToHost", "MemcpyFromHost")
                or len(node.inputs) != 1
                or len(node.outputs) != 1
                or node.inputs[0] is None
            ):
                continue

            src = node.inputs[0]
            memcpy_out = node.outputs[0]
            if memcpy_out in graph.outputs:
                # Memcpy on the output boundary: preserve the public output name by
                # moving it onto the upstream producer's value.
                output_name = memcpy_out.name
                memcpy_out.replace_all_uses_with(src, replace_graph_outputs=True)
                src.name = output_name
            else:
                memcpy_out.replace_all_uses_with(src)
            graph.remove(node, safe=True)
            removed += 1
        return removed


class RenameInputDims(Surgeon):
    """Rename / promote a dimension in an input tensor's shape to a named symbolic dim.

    This surgery replaces a concrete dim_value (e.g. ``1``) with a symbolic
    dim_param string (e.g. ``"num_images"``).  Useful when torch.export
    specialises a batch-like input dimension to a concrete value because its
    shape is algebraically derived from another symbolic dimension, yet ONNX
    Runtime must accept a variable-length tensor at inference time.

    Specify the target input either by name (preferred) or by index.

    Example usage:
        {
            "surgeon": "RenameInputDims",
            "input_name": "image_grid_thw",
            "dim_idx": 0,
            "dim_name": "num_images"
        }
    """

    def __init__(
        self,
        dim_idx: int,
        dim_name: str,
        input_name: str | None = None,
        input_idx: int | None = None,
    ):
        super().__init__()
        if input_name is None and input_idx is None:
            raise ValueError("Either 'input_name' or 'input_idx' must be provided.")
        self.input_name = input_name
        self.input_idx = input_idx
        self.dim_idx = dim_idx
        self.dim_name = dim_name

    def call_ir(self, model: ir.Model) -> ir.Model:
        inputs = list(model.graph.inputs)

        if self.input_name is not None:
            target = next((v for v in inputs if v.name == self.input_name), None)
            if target is None:
                available = [v.name for v in inputs]
                raise ValueError(f"Input '{self.input_name}' not found in graph. Available inputs: {available}")
        else:
            if self.input_idx >= len(inputs):
                raise ValueError(f"input_idx {self.input_idx} is out of range. Model has {len(inputs)} inputs.")
            target = inputs[self.input_idx]

        if target.shape is None:
            raise ValueError(f"Input '{target.name}' has no shape information; cannot rename dimensions.")

        if self.dim_idx >= len(target.shape):
            raise ValueError(
                f"dim_idx {self.dim_idx} is out of range. Input '{target.name}' has {len(target.shape)} dimensions."
            )

        new_dims = list(target.shape)
        new_dims[self.dim_idx] = self.dim_name
        target.shape = ir.Shape(new_dims)
        return model


class GraphSurgeries(Pass):
    """ONNX graph surgeries collections.

    This pass applies a list of surgeries to the ONNX model.
    Each surgery is a transformation on the ONNX graph.

    Example:
        surgeries: {
            type: "GraphSurgeries",
            surgeries: [
                {
                    "surgeon": "RenameInputs",
                    "old_names": ["input1", "input2"]
                    "new_names": ["renamed_input1", "renamed_input2"]
                },
                {
                    "surgeon": "RenameOutputs",
                    "old_names": ["output1", "output2"]
                    "new_names": ["renamed_output1", "renamed_output2"]
                }
            ]
        }

    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "surgeries": PassConfigParam(
                type_=list[dict[str, Any]],
                default_value=[],
                required=True,
                description="List of surgeries to apply, each with its type and parameters",
            ),
            "remove_duplicate_initializers": PassConfigParam(
                type_=bool,
                default_value=True,
                description="""
                   Apply DeduplicateHashedInitializersPass after graph surgeries in case graph surgeries add duplicated initializers
                """,
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        surgeries = config.surgeries
        onnx_model = model.load_model()
        for surgery in surgeries:
            logger.info("Applying surgery: %s", surgery)
            surgeon_instance = self.init_surgeon_instance(surgery)
            onnx_model = surgeon_instance(onnx_model)

        if config.remove_duplicate_initializers:
            deduped_model = DeduplicateHashedInitializersPass()(ir.from_proto(onnx_model)).model
            return model_proto_to_olive_model(ir.to_proto(deduped_model), output_model_path, config)
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def init_surgeon_instance(self, surgery):
        surgeon_name = surgery.get("surgeon").lower()
        if not surgeon_name:
            raise ValueError("Surgeon is not specified")

        surgeon_class = Surgeon.registry.get(surgeon_name)
        if not surgeon_class:
            raise ValueError(f"Surgeon '{surgeon_name}' does not exist. Available surgeons: {Surgeon.registry.keys()}")

        required_params, optional_params = self.get_surgeon_parameters(surgeon_class)
        provided_params = set(surgery.keys()) - {"surgeon"}
        missing_params = set(required_params) - provided_params
        extra_params = provided_params - set(required_params) - set(optional_params)

        if missing_params:
            raise ValueError(f"Missing parameters for surgery '{surgeon_name}': {missing_params}")
        if extra_params:
            logger.warning("Ignoring extra parameters for surgery '%s': %s", surgeon_name, extra_params)

        init_params = {param: surgery[param] for param in required_params}
        init_params.update({param: surgery[param] for param in optional_params if param in surgery})
        return surgeon_class(**init_params)

    @staticmethod
    def get_surgeon_parameters(surgeon_class: type[Surgeon]) -> tuple[list[str], list[str]]:
        parameters = inspect.signature(surgeon_class.__init__).parameters

        positional_args = [
            name
            for name, param in parameters.items()
            if param.default == param.empty and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        positional_args.remove("self")
        keyword_args = [
            name
            for name, param in parameters.items()
            if param.default != param.empty or param.kind == param.KEYWORD_ONLY
        ]
        return positional_args, keyword_args
