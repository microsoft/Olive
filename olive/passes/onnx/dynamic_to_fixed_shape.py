# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Any, Callable

import onnx_ir as ir
from pydantic import model_validator

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import (
    get_external_data_config,
    ir_model_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _iter_shaped_values(graph: ir.Graph):
    """Yield all value objects in a single graph that may carry shape information."""
    yield from graph.inputs
    for node in graph:
        yield from node.outputs
    yield from graph.outputs


def _make_dim_param_fixed(ir_model: ir.Model, param_name: str, value: int) -> None:
    """Replace every occurrence of the symbolic dim ``param_name`` with ``value`` across the model.

    Mirrors onnxruntime.tools.onnx_model_utils.make_dim_param_fixed but operates on an ir.Model,
    including subgraphs.
    """
    for graph in ir_model.graphs():
        for val in _iter_shaped_values(graph):
            if val is None or val.shape is None:
                continue
            dims = list(val.shape)
            changed = False
            for idx, dim in enumerate(dims):
                if isinstance(dim, ir.SymbolicDim) and dim.value == param_name:
                    dims[idx] = value
                    changed = True
            if changed:
                val.shape = ir.Shape(dims)


def _remove_invalid_dim_values(ir_model: ir.Model) -> None:
    """Unset any fixed dim values that are less than 1 (typically -1 placeholders for dynamic dims)."""
    for graph in ir_model.graphs():
        for val in _iter_shaped_values(graph):
            if val is None or val.shape is None:
                continue
            dims = list(val.shape)
            changed = False
            for idx, dim in enumerate(dims):
                if isinstance(dim, int) and dim < 1:
                    dims[idx] = None
                    changed = True
            if changed:
                val.shape = ir.Shape(dims)


def _make_input_shape_fixed(ir_model: ir.Model, input_name: str, fixed_shape: list[int]) -> None:
    """Set the shape of the named graph input to ``fixed_shape``.

    Mirrors onnxruntime.tools.onnx_model_utils.make_input_shape_fixed but operates on an ir.Model.
    """
    # remove any invalid dim values first. typically this is a dim_value of -1.
    _remove_invalid_dim_values(ir_model)

    for graph_input in ir_model.graph.inputs:
        if graph_input.name != input_name:
            continue

        # graph inputs are required to have a shape to provide the rank
        if graph_input.shape is None:
            raise ValueError(f"Input {input_name} does not have a shape")

        dims = list(graph_input.shape)
        if len(dims) != len(fixed_shape):
            raise ValueError(f"Rank mismatch. Existing:{len(dims)} Replacement:{len(fixed_shape)}")

        new_dims = list(dims)
        for idx, dim in enumerate(dims):
            if isinstance(dim, int):
                # check any existing fixed dims match
                if dim != fixed_shape[idx]:
                    raise ValueError(
                        f"Can't replace existing fixed size of {dim} with {fixed_shape[idx]} for dimension {idx + 1}"
                    )
            elif isinstance(dim, ir.SymbolicDim) and dim.value is not None:
                # replacing a dim_param so have to do that through the entire model
                _make_dim_param_fixed(ir_model, dim.value, fixed_shape[idx])
                new_dims[idx] = fixed_shape[idx]
            else:
                # replacing an unknown dim
                new_dims[idx] = fixed_shape[idx]

        graph_input.shape = ir.Shape(new_dims)
        return

    valid_names = ",".join(i.name for i in ir_model.graph.inputs if i.name)
    raise ValueError(f"Input {input_name} was not found in graph inputs. Valid input names are: {valid_names}")


def _fix_output_shapes(ir_model: ir.Model) -> None:
    """Run shape inference on the model and update graph output shapes to make them fixed."""
    from onnxruntime.tools.onnx_model_utils import is_fixed_size_tensor
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    # use the onnxruntime shape inference tool since it can handle large models as well as contrib ops
    model_proto = ir.to_proto(ir_model)
    inferred_proto = SymbolicShapeInference.infer_shapes(model_proto, auto_merge=True, guess_output_rank=True)
    inferred_outputs = {o.name: o for o in inferred_proto.graph.output}

    for output in ir_model.graph.outputs:
        if output is None or output.name is None or _shape_is_fixed(output):
            continue
        new_o = inferred_outputs.get(output.name)
        if new_o is not None and is_fixed_size_tensor(new_o):
            output.shape = ir.Shape([dim.dim_value for dim in new_o.type.tensor_type.shape.dim])


def _shape_is_fixed(value: ir.Value) -> bool:
    """Return True if the value has a shape where every dimension is a fixed positive integer."""
    if value is None or value.shape is None:
        return False
    return all(isinstance(dim, int) and dim > 0 for dim in value.shape)


def fix_dim_params(ir_model: ir.Model, dim_params: list[str], dim_values: list[int]) -> None:
    """Fix the dimension parameters in an ir.Model.

    :param dim_params: The dimension parameters to fix.
    :param dim_values: The values to set for the dimension parameters.
    """
    dim_params = list(dim_params)
    dim_values = list(dim_values)
    assert len(dim_params) == len(dim_values), "dim_params and dim_values must have the same number of elements."
    assert all(i >= 0 for i in dim_values), "dim_values must be all >= 0"

    for param, value in zip(dim_params, dim_values):
        _make_dim_param_fixed(ir_model, param, value)

    # update the output shapes to make them fixed
    _fix_output_shapes(ir_model)


def fix_input_shapes(ir_model: ir.Model, input_names: list[str], input_shapes: list[list[int]]) -> None:
    """Fix the input shapes in an ir.Model.

    :param input_names: The input names to fix.
    :param input_shapes: The shapes to set for the inputs.
    """
    assert len(input_names) == len(input_shapes), "input_names and input_shapes must have the same number of elements."
    assert all(all(i > 0 for i in shape) for shape in input_shapes), "input_shapes must be all > 0"

    for name, shape in zip(input_names, input_shapes):
        _make_input_shape_fixed(ir_model, name, shape)

    # update the output shapes to make them fixed
    _fix_output_shapes(ir_model)


class DynamicToFixedShape(Pass):
    """Convert dynamic shape to fixed shape for ONNX model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "dim_param": PassConfigParam(
                type_=list[str],
                default_value=None,
                required=False,
                description="Symbolic parameter name. Provide dim_value if specified.",
            ),
            "dim_value": PassConfigParam(
                type_=list[int],
                default_value=None,
                required=False,
                description="Value to replace dim_param with in the model. Must be > 0.",
            ),
            "input_name": PassConfigParam(
                type_=list[str],
                default_value=None,
                required=False,
                description="Model input name to replace shape of. Provide input_shape if specified.",
            ),
            "input_shape": PassConfigParam(
                type_=list[list[int]],
                default_value=None,
                required=False,
                description=(
                    "Shape to use for input_shape. Provide comma separated list for the shape. "
                    "All values must be > 0. e.g. [1,3,256,256]"
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    @classmethod
    def _validators(cls) -> dict[str, Callable[..., Any]]:
        return {
            "validate_configs": model_validator(mode="before")(_jointly_validate_configs),
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        ir_model = model.load_ir_model()
        ir.external_data.load_to_model(ir_model)
        output_model_path = resolve_onnx_path(output_model_path)

        if config.dim_param:
            fix_dim_params(ir_model, config.dim_param, config.dim_value)
        elif config.input_name:
            fix_input_shapes(ir_model, config.input_name, config.input_shape)

        return ir_model_to_olive_model(ir_model, output_model_path, config)


def _jointly_validate_configs(cls, values):
    if values.get("input_name") and values.get("dim_param"):
        raise ValueError("Cannot set both dim_param and input_name at the same time.")
    if not values.get("input_name") and not values.get("dim_param"):
        raise ValueError("dim_param and input_name cannot be both empty.")

    # cannot use if values["dim_param"] ^ values["dim_value"] because the value could be list
    # and list cannot be used in xor operation
    if (not values.get("dim_param")) ^ (not values.get("dim_value")):
        raise ValueError("dim_param and dim_value must be both provided or both None.")
    if (not values.get("input_name")) ^ (not values.get("input_shape")):
        raise ValueError("input_name and input_shape must be both provided or both None.")

    if values.get("dim_param") and values.get("dim_value"):
        if len(values["dim_param"]) != len(values["dim_value"]):
            raise ValueError("dim_param and dim_value must have the same number of elements.")
        if any(i < 0 for i in values["dim_value"]):
            raise ValueError("dim_value must be all >= 0 when dim_param is provided.")

    if values.get("input_name") and values.get("input_shape"):
        if len(values["input_name"]) != len(values["input_shape"]):
            raise ValueError("input_name and input_shape must have the same number of elements.")
        if any(any(i <= 0 for i in shape) for shape in values["input_shape"]):
            raise ValueError("input_shape must be all > 0 when input_name is provided.")
    return values
