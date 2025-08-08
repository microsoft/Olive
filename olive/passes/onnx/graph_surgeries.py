# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
import inspect
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import onnx
from onnx import ModelProto, TensorProto
from onnx.helper import make_tensor
from onnxscript import ir, rewriter
from onnxscript.rewriter import pattern

from olive.constants import MSFT_DOMAIN
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


# pylint: disable=W0621


class Surgeon:
    """Base class for surgeons that operate on the ONNX IR model."""

    # Refer to https://microsoft.github.io/onnxscript/intermediate_representation/ir_api.html#onnxscript.ir.Model
    # for the IR model API.

    registry: ClassVar[dict[str, type["Surgeon"]]] = {}

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


class InferShapes(ProtoSurgeon):
    def __call__(self, model: ModelProto):
        return onnx.shape_inference.infer_shapes(model)


class RemoveShapes(ProtoSurgeon):
    def __call__(self, model: ModelProto):
        while len(model.graph.value_info) > 0:
            model.graph.value_info.pop()
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


class ReplaceErfWithTanh(ProtoSurgeon):
    DTYPE_MAP: Mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.BFLOAT16: np.uint16,
        TensorProto.INT8: np.int8,
        TensorProto.INT16: np.int16,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.UINT8: np.uint8,
        TensorProto.UINT16: np.uint16,
        TensorProto.UINT32: np.uint32,
        TensorProto.UINT64: np.uint64,
    }

    def __call__(self, model: ModelProto):
        idx = 0
        while idx < len(model.graph.node):
            node = model.graph.node[idx]
            if node.op_type == "Erf":
                inputs = node.input
                outputs = node.output
                input_dtype = self._get_input_dtype(model, inputs[0])
                np_type = self.DTYPE_MAP.get(input_dtype)
                if np_type is None:
                    logger.warning(
                        "Unsupported dtype %s for node %s. Skip replacing Erf with Tanh.", input_dtype, node.name
                    )
                    idx += 1
                    continue

                model.graph.node.remove(node)
                name = f"scale_{idx}"
                output_scale = f"mul_{idx}"

                # scaling constant for tanh
                value = np.array(605 / 503, dtype=np_type)
                scale = onnx.helper.make_tensor(
                    name=name,
                    data_type=input_dtype,
                    dims=value.shape,
                    vals=value.flatten().tolist(),
                )
                model.graph.initializer.append(scale)

                mul_node = onnx.helper.make_node(
                    "Mul", inputs=[inputs[0], name], outputs=[output_scale], name=f"Sub_Mul_{idx}"
                )
                tanh_node = onnx.helper.make_node(
                    "Tanh", inputs=[output_scale], outputs=outputs, name=f"Sub_Tanh_{idx}"
                )

                model.graph.node.insert(idx, mul_node)
                model.graph.node.insert(idx + 1, tanh_node)
                idx += 2
            else:
                idx += 1
        return model

    def _get_input_dtype(self, model, name):
        for inp in model.graph.input:
            if inp.name == name:
                return inp.type.tensor_type.elem_type
        for vi in model.graph.value_info:
            if vi.name == name:
                return vi.type.tensor_type.elem_type
        for init in model.graph.initializer:
            if init.name == name:
                return init.data_type
        raise ValueError(f"Cannot find dtype for {name}")


class ZeroOutInput(ProtoSurgeon):
    def __init__(self, node_name, input_idx):
        self.node_name = node_name
        self.input_idx = input_idx

    def __call__(self, model: ModelProto):
        from onnx.helper import make_node
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        node = self.get_node_by_name(model, self.node_name)
        if node is None:
            logger.warning("Node %s not found in the model.", self.node_name)
            return model

        input_name = node.input[self.input_idx]

        target_shape = None
        target_type = None

        shapes = self.get_tensor_shapes(model)
        types = self.get_tensor_types(model)

        target_node = self.get_node_by_name(model, input_name, True)
        if target_node is not None:
            if input_name in shapes and input_name in types:
                target_shape = shapes[input_name]
                target_type = types[input_name]
            else:
                logger.warning("Cannot determine shape and type for input '%s'.", input_name)
                return model

        elif input_name in {inp.name for inp in model.graph.input}:
            target = next(inp for inp in model.graph.input if inp.name == input_name)
            if target.type.tensor_type.shape.dim:
                target_shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in target.type.tensor_type.shape.dim]
                target_type = target.type.tensor_type.elem_type
            else:
                logger.warning("Cannot determine shape and type for input '%s'.", input_name)
                return model

        elif input_name in {init.name for init in model.graph.initializer}:
            target = next(init for init in model.graph.initializer if init.name == input_name)
            target_shape = target.dims
            target_type = target.data_type
        else:
            logger.warning("Input '%s' not found in the model.", input_name)
            return model

        if target_shape is None or target_type is None:
            raise ValueError(f"Cannot determine shape and type for input '{input_name}'.")

        zero_values = np.zeros(target_shape, dtype=TENSOR_TYPE_TO_NP_TYPE[target_type])

        zero_tensor = make_tensor(
            name=f"{self.node_name}_zero_tensor",
            data_type=target_type,
            dims=target_shape,
            vals=zero_values.flatten().tolist(),
        )

        zero_node = make_node(
            op_type="Constant",
            inputs=[],
            outputs=[f"{self.node_name}_zero_output_0"],
            name=f"{self.node_name}_zero",
            value=zero_tensor,
        )

        model.graph.node.append(zero_node)
        node.input[self.input_idx] = zero_node.output[0]

        return model


class RemoveInputs(ProtoSurgeon):
    def __init__(self, names):
        self.names = names

    def __call__(self, model: ModelProto):
        for name in self.names:
            for graph_input in model.graph.input:
                if graph_input.name == name:
                    model.graph.input.remove(graph_input)
                    break

        nodes_to_remove = []

        for node in model.graph.node:
            node.input[:] = [input_name for input_name in node.input if input_name not in self.names]
            if len(node.input) == 0:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            model.graph.node.remove(node)

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


class RMSNormToL2Norm(ProtoSurgeon):
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

    def __call__(self, model: ModelProto):
        dag = OnnxDAG(model)

        modified = 0
        removed_nodes = set()
        replaced_initializers = set()
        for node_name in dag.get_node_names():
            if node_name in removed_nodes or dag.get_node_op_type(node_name) != "Pow":
                continue

            rmsnorm_nodes = self.get_rmsnorm_nodes(node_name, dag)
            if not rmsnorm_nodes:
                continue

            graph_idx = dag.get_graph_idx(node_name)

            # name for the new L2Norm node
            l2norm_node_name = self.create_new_name(node_name, "Pow", "L2Norm")
            l2norm_node_output_name = f"{l2norm_node_name}_output_0"

            # create L2Norm node
            l2norm_node = onnx.helper.make_node(
                "LpNormalization",
                inputs=[dag.get_node_inputs(node_name)[0]],
                outputs=[l2norm_node_output_name],
                name=l2norm_node_name,
                axis=-1,
                p=2,
            )

            # Muliply weight by sqrt(N)
            final_node_name = rmsnorm_nodes[-1]
            final_node_children = dag.get_consumers(final_node_name)
            # can be Cast or Mul
            if len(final_node_children) != 1 or dag.get_node_op_type(final_node_children[0]) not in ["Cast", "Mul"]:
                logger.debug("RMSNorm Pattern does not end with Cast or Mul. Found %s", final_node_children)
                continue
            final_node_output_name = dag.get_node_outputs(final_node_name)[0]
            # add value info for the new L2Norm node output
            if final_node_output_vi := dag.get_value_info_proto(final_node_output_name):
                l2norm_node_output_vi = onnx.ValueInfoProto()
                l2norm_node_output_vi.CopyFrom(final_node_output_vi)
                l2norm_node_output_vi.name = l2norm_node_output_name
                dag.add_value_info(l2norm_node_output_vi, graph_idx)

            # Get the weight Mul node
            if dag.get_node_op_type(final_node_children[0]) == "Mul":
                rmsnorm_mul_node_name = final_node_children[0]
            else:
                cast_node_children = dag.get_consumers(final_node_children[0])
                if len(cast_node_children) != 1 or dag.get_node_op_type(cast_node_children[0]) != "Mul":
                    logger.debug("RMSNorm Pattern does not end with Cast -> Mul. Found %s", cast_node_children)
                    continue
                rmsnorm_mul_node_name = cast_node_children[0]

            # Get the weight Mul node inputs
            rmsnorm_weight_name = None
            for input_name in dag.get_node_inputs(rmsnorm_mul_node_name):
                if dag.is_initializer(input_name):
                    rmsnorm_weight_name = input_name
                    break
            if rmsnorm_weight_name is None:
                logger.debug("RMSNorm Mul node does not have initializer input")
                continue

            # update weight and replace initializer
            if rmsnorm_weight_name not in replaced_initializers:
                # rotated models have all 1s and might share initializer
                # don't want to multiply by sqrt(N) multiple times even though it is fine in all 1s case
                rmsnorm_weight = dag.get_initializer_np_array(rmsnorm_weight_name)
                sqrt_n = np.sqrt(rmsnorm_weight.shape[-1]).astype(rmsnorm_weight.dtype)
                if np.all(rmsnorm_weight == 1):
                    # this is possible in a quarot/spinquant rotated model
                    # Multiplying by 1D is probably faster
                    rmsnorm_weight = np.array([1], dtype=rmsnorm_weight.dtype)
                rmsnorm_weight = sqrt_n * rmsnorm_weight

                dag.replace_initializer(onnx.numpy_helper.from_array(rmsnorm_weight, name=rmsnorm_weight_name))
                replaced_initializers.add(rmsnorm_weight_name)

            # add and replace nodes
            dag.add_node(l2norm_node, graph_idx)
            dag.replace_node_input(final_node_children[0], final_node_output_name, l2norm_node_output_name)
            for rms_node_name in rmsnorm_nodes[::-1]:
                dag.remove_node(rms_node_name)
                removed_nodes.add(rms_node_name)

            modified += 1

        if modified > 0:
            logger.debug("Replaced %d RMSNorm nodes with L2Norm nodes", modified)

        dag.update()
        return dag.model

    @staticmethod
    def get_rmsnorm_nodes(pow_node: str, dag: OnnxDAG) -> Optional[list[str]]:
        # Two possible patterns:
        # x / sqrt(mean(x^2) + epsilon): Pow -> ReduceMean -> Add -> Sqrt -> Div
        # x * 1 / sqrt(mean(x^2) + epsilon): Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul
        pattern = ["Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul"]
        pow_node_input = dag.get_node_inputs(pow_node)[0]
        current_node = pow_node
        rmsnorm_nodes = [current_node]
        for op_type in pattern[1:]:
            child_nodes = dag.get_consumers(current_node)
            if len(child_nodes) != 1 or dag.get_node_op_type(child_nodes[0]) != op_type:
                return []
            current_node = child_nodes[0]
            rmsnorm_nodes.append(current_node)
            if pow_node_input in dag.get_node_inputs(current_node):
                # this can happen either at Div or Mul
                # early stopping if it is Div
                break

        return rmsnorm_nodes if len(rmsnorm_nodes) >= (len(pattern) - 1) else []


class SimplifiedLayerNormToL2Norm(ProtoSurgeon):
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

    def __call__(self, model: ModelProto):
        dag = OnnxDAG(model)

        modified = 0
        for node_name in dag.get_node_names():
            op_type = dag.get_node_op_type(node_name)
            if op_type not in {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}:
                continue

            if len(dag.get_node_inputs(node_name, True)) != 2 + int(op_type == "SkipSimplifiedLayerNormalization"):
                # SimplifiedLayerNormalization: X, scale supported
                # SkipSimplifiedLayerNormalization: input, skip, gamma supported
                continue
            if len(dag.get_node_outputs(node_name, True)) > 1 + int(op_type == "SkipSimplifiedLayerNormalization"):
                # SimplifiedLayerNormalization: output supported
                # SkipSimplifiedLayerNormalization: output, input_skip_bias_sum (optional) supported
                continue

            graph_idx = dag.get_graph_idx(node_name)

            if op_type == "SkipSimplifiedLayerNormalization":
                # SimplifiedLayerNormalization preceded by an Add node
                add_node_name = self.create_new_name(node_name, op_type, "Add")
                add_node_output_name = f"{add_node_name}_output_0"
                dag.add_node(
                    onnx.helper.make_node(
                        "Add",
                        inputs=dag.get_node_inputs(node_name, True)[:2],
                        outputs=[add_node_output_name],
                        name=add_node_name,
                    ),
                    graph_idx,
                )

                # input_skip_bias_sum is used downstream
                if len(dag.get_node_outputs(node_name, True)) == 2:
                    skip_output_name = dag.get_node_outputs(node_name, True)[1]
                    if skip_output_vi := dag.get_value_info_proto(skip_output_name):
                        add_output_vi = onnx.ValueInfoProto()
                        add_output_vi.CopyFrom(skip_output_vi)
                        add_output_vi.name = add_node_output_name
                        dag.add_value_info(add_output_vi, graph_idx)

                    for child_name in dag.get_consumers(skip_output_name):
                        dag.replace_node_input(child_name, skip_output_name, add_node_output_name)

                l2norm_node_input_name = add_node_output_name
            else:
                l2norm_node_input_name = dag.get_node_inputs(node_name, True)[0]

            layernorm_node_output_name = dag.get_node_outputs(node_name, True)[0]

            # add L2Norm node
            l2norm_node_name = self.create_new_name(node_name, op_type, "L2Norm")
            l2norm_node_output_name = f"{l2norm_node_name}_output_0"
            dag.add_node(
                onnx.helper.make_node(
                    "LpNormalization",
                    inputs=[l2norm_node_input_name],
                    outputs=[l2norm_node_output_name],
                    name=l2norm_node_name,
                    axis=-1,
                    p=2,
                ),
                graph_idx,
            )
            if layernorm_node_output_vi := dag.get_value_info_proto(layernorm_node_output_name):
                l2norm_node_output_vi = onnx.ValueInfoProto()
                l2norm_node_output_vi.CopyFrom(layernorm_node_output_vi)
                l2norm_node_output_vi.name = l2norm_node_output_name
                dag.add_value_info(l2norm_node_output_vi, graph_idx)

            # add Mul node
            mul_weight_name = dag.get_node_inputs(node_name, True)[-1]
            mul_weight = dag.get_initializer_np_array(mul_weight_name)
            sqrt_n = np.sqrt(mul_weight.shape[-1]).astype(mul_weight.dtype)
            if np.all(mul_weight == 1):
                # this is possible in a quarot/spinquant rotated model
                # Multiplying by 1D is probably faster
                mul_weight = np.array([1], dtype=mul_weight.dtype)
            mul_weight = sqrt_n * mul_weight
            dag.replace_initializer(onnx.numpy_helper.from_array(mul_weight, name=mul_weight_name))

            mul_node_name = self.create_new_name(node_name, op_type, "Mul")
            mul_node_output_name = f"{mul_node_name}_output_0"
            dag.add_node(
                onnx.helper.make_node(
                    "Mul",
                    inputs=[l2norm_node_output_name, mul_weight_name],
                    outputs=[mul_node_output_name],
                    name=mul_node_name,
                ),
                graph_idx,
            )
            if layernorm_node_output_vi := dag.get_value_info_proto(layernorm_node_output_name):
                mul_output_vi = onnx.ValueInfoProto()
                mul_output_vi.CopyFrom(layernorm_node_output_vi)
                mul_output_vi.name = mul_node_output_name
                dag.add_value_info(mul_output_vi, graph_idx)

            for child_name in dag.get_consumers(layernorm_node_output_name):
                dag.replace_node_input(child_name, layernorm_node_output_name, mul_node_output_name)

            # remove node
            dag.remove_node(node_name)

            modified += 1

        if modified > 0:
            logger.debug("Replaced %d Skip/SimplifiedLayerNormalization nodes with L2Norm nodes", modified)

        dag.update()
        return dag.model


class PowReduceSumPowDiv2LpNorm(ProtoSurgeon):
    """Merge Pow ReduceSum Pow Div pattern to L2Norm.

    Below pattern is replaced with LpNormalization (p=2, axis=-1):
    [Root] --> Pow --> ReduceSum --> Pow -- Div
        |                                    |
        +------------------------------------+
    """

    def find_lp_norm_pattern(self, model: ModelProto):
        matches = []
        for node in model.graph.node:
            if node.op_type != "Div":
                continue

            div_node = node
            pow2_node = None
            sum_node = None
            pow_half_node = None

            # Find Pow node input to Div (norm)
            norm_input = div_node.input[1]
            for node2 in model.graph.node:
                if node2.output[0] == norm_input and node2.op_type == "Pow":
                    pow_half_node = node2
                    break

            if not pow_half_node:
                continue

            sum_input = pow_half_node.input[0]
            for node3 in model.graph.node:
                if node3.output[0] == sum_input and node3.op_type == "ReduceSum":
                    sum_node = node3
                    break

            if not sum_node:
                continue

            pow_input = sum_node.input[0]
            for node4 in model.graph.node:
                if node4.output[0] == pow_input and node4.op_type == "Pow":
                    pow2_node = node4
                    break

            if not pow2_node:
                continue

            # Confirm constant exponents: 2 and 0.5
            def is_const_pow(node, exp_val):
                return node.op_type == "Pow" and exp_val in [
                    onnx.numpy_helper.to_array(init).item()
                    for init in model.graph.initializer
                    if init.name == node.input[1]
                ]

            if not (is_const_pow(pow2_node, 2.0) and is_const_pow(pow_half_node, 0.5)):
                continue

            # Save matched nodes
            matches.append((div_node, pow_half_node, sum_node, pow2_node))

        return matches

    def replace_with_lp_normalization(self, model: ModelProto):
        matches = self.find_lp_norm_pattern(model)
        logger.info("Replacing %d instance of the pattern with LpNorm", len(matches))
        for div_node, pow_half_node, sum_node, pow2_node in matches:
            input_name = pow2_node.input[0]
            output_name = div_node.output[0]

            lp_node = onnx.helper.make_node(
                "LpNormalization",
                inputs=[input_name],
                outputs=[output_name],
                name=div_node.name + "_lp_norm",
                p=2,
                axis=-1,
            )

            # Remove old nodes
            nodes_to_remove = {div_node.name, pow_half_node.name, sum_node.name, pow2_node.name}
            nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]

            # Insert new node
            model.graph.ClearField("node")
            model.graph.node.extend(nodes)
            model.graph.node.append(lp_node)

        return model

    def __call__(self, model: ModelProto):
        dag = OnnxDAG(self.replace_with_lp_normalization(model))
        dag.update()
        return dag.model


class MatMulAddToGemm(ProtoSurgeon):
    """Replace MatMul + Add with Gemm.

    Second MatMul input must be a 2D tensor and the other input of the Add node must be a 1D tensor.
    If the first MatMul input is more than 2D and the shapes are static, it is reshaped to 2D before the Gemm
    node and reshaped back to the original shape after the Gemm node.
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
            add_output = dag.get_node_outputs(add_name)[0]

            gemm_name = self.create_new_name(matmul_name, "MatMul", "Gemm")
            gemm_inputs = [*matmul_inputs, bias_input]

            matmul_a_shape = matmul_input_shapes[0]
            gemm_output_shape = matmul_output_shape
            if len(matmul_a_shape) != 2:
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

            if len(matmul_a_shape) != 2:
                # need to reshape the output to original shape
                post_reshape_name = self.create_new_name(gemm_name, "Gemm", "Reshape_post")
                final_output_name = self.add_reshape_node(
                    dag,
                    graph_idx,
                    post_reshape_name,
                    gemm_output_name,
                    matmul_output_shape,
                    elem_type,
                )

            # point all of the consumers of the original add to the final output
            for consumer in dag.get_consumers(add_name):
                dag.replace_node_input(consumer, add_output, final_output_name)

            # remove the original add and matmul nodes
            for to_remove in [add_name, matmul_name]:
                dag.remove_node(to_remove)
                removed_nodes.add(to_remove)
            modified += 1

        if modified > 0:
            logger.debug("Replaced %d MatMul + Add nodes with Gemm nodes", modified)

        dag.update()
        return dag.model

    @staticmethod
    def add_reshape_node(
        dag: OnnxDAG, graph_idx: int, node_name: str, input_name: str, target_shape: list[int], output_elem_type: int
    ) -> str:
        """Add a reshape node to the graph.

        :param dag: The OnnxDAG object.
        :param graph_idx: The index of the graph.
        :param node_name: The name of the node.
        :param input_name: The name of the input tensor.
        :param target_shape: The target shape for the reshape operation.
        :param output_elem_type: The element type of the output tensor.
        :return: The name of the output tensor after reshaping.
        """
        # need to reshape the first input to 2D
        reshape_shape_name = f"{node_name}_shape"
        reshape_output_name = f"{node_name}_output"
        dag.add_initializer(
            onnx.numpy_helper.from_array(np.array(target_shape, dtype=np.int64), reshape_shape_name), graph_idx
        )
        dag.add_node(
            onnx.helper.make_node(
                "Reshape",
                [input_name, reshape_shape_name],
                [reshape_output_name],
                name=node_name,
            ),
            graph_idx,
        )
        dag.add_value_info(
            onnx.helper.make_tensor_value_info(
                reshape_output_name,
                output_elem_type,
                target_shape,
            ),
            graph_idx,
        )
        return reshape_output_name


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
        if dag.get_node_op_type(first_node) == "GroupQueryAttention" and len(first_node_inputs) != 9:
            return dag.model

        # check if cos_cache and sin_cache come from an If node
        cache_names = {"cos_cache": first_node_inputs[-2], "sin_cache": first_node_inputs[-1]}
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


class ReplaceAttentionMaskValue(ProtoSurgeon):
    """Replace the value of extended attention mask with a new value.

    This surgery is useful if the default mask value does not quantize well due to numerical instability.
    """

    ALLOWED_CONSUMER_OPS: ClassVar[set[str]] = {"Add", "Mul", "Expand", "Where"}

    def __init__(self, threshold: float = -3e30, replacement: float = -1e4):
        self.threshold = threshold
        self.replacement = replacement

    def __call__(self, model: ModelProto):
        dag = OnnxDAG(model)
        modified = 0

        # update any constant or constantofshape nodes with the threshold value
        for node_name in dag.get_node_names():
            op_type = dag.get_node_op_type(node_name)
            node_proto = dag.get_node_proto(node_name)
            if not (
                op_type in {"Constant", "ConstantOfShape"}
                and node_proto.attribute
                and node_proto.attribute[0].t
                and node_proto.attribute[0].t.data_type == onnx.TensorProto.FLOAT
                and self.valid_consumers(node_name, dag)
            ):
                continue

            value = onnx.helper.get_attribute_value(node_proto.attribute[0])
            tensor_value = onnx.numpy_helper.to_array(value)
            if (tensor_value_new := self.new_tensor_value(tensor_value)) is not None:
                node_proto.ClearField("attribute")
                node_proto.attribute.extend(
                    [onnx.helper.make_attribute("value", onnx.numpy_helper.from_array(tensor_value_new))]
                )
                modified += 1

        # update any initializer nodes with the threshold value
        for init_name in dag.get_initializer_names():
            init_proto = dag.get_initializer_proto(init_name)
            if not (init_proto.data_type == onnx.TensorProto.FLOAT and self.valid_consumers(init_name, dag)):
                continue

            tensor_value = onnx.numpy_helper.to_array(init_proto)
            if (tensor_value_new := self.new_tensor_value(tensor_value)) is not None:
                dag.replace_initializer(onnx.numpy_helper.from_array(tensor_value_new, name=init_name))
                modified += 1

        if modified > 0:
            logger.debug("Replaced %d values below threshold with replacement.", modified)

        dag.update()
        return dag.model

    def valid_consumers(self, name: str, dag: OnnxDAG) -> bool:
        """Check if the consumers of the node are valid.

        This is to prevent checking the tensor value of large tensors unnecessarily.
        """
        for consumer in dag.get_consumers(name):
            if dag.get_node_op_type(consumer) not in ReplaceAttentionMaskValue.ALLOWED_CONSUMER_OPS:
                return False
        return True

    def new_tensor_value(self, tensor_value: np.ndarray) -> Optional[np.ndarray]:
        """Replace values below the threshold with the replacement value."""
        if np.any(tensor_value < self.threshold):
            tensor_value_new = tensor_value.copy()
            tensor_value_new[tensor_value_new < self.threshold] = self.replacement
            return tensor_value_new
        return None


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
