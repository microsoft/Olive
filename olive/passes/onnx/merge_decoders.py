# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Implementation derived from decoder merge logic in Optimum.
# https://github.com/huggingface/optimum/blob/main/optimum/onnx/graph_transformations.py

import hashlib
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import onnx
from onnx import GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto

if TYPE_CHECKING:
    from numpy.typing import NDArray

# pylint: disable=consider-using-enumerate

logger = logging.getLogger(__name__)


def _get_onnx_opset(model: ModelProto) -> int:
    opset_import = model.opset_import[0]
    return opset_import.version


def _infer_output_shape(output: ValueInfoProto) -> List[Any]:
    output_shape: List[Any] = []
    for dim in output.type.tensor_type.shape.dim:
        if hasattr(dim, "dim_param"):
            output_shape.append(dim.dim_param)
        elif hasattr(dim, "dim_value"):
            output_shape.append(dim.dim_value)
        else:
            raise ValueError("Cannot find `dim_param` nor `dim_value` in the output dimension info.")

    return output_shape


def _unify_onnx_outputs(model1: ModelProto, model2: ModelProto, strict: bool):
    """Unifies the outputs of two ONNX model protos.

    The outputs of model1 will be replaced by outputs of model2.
    According to the rules of "If" op, two subgraphs must have the same number of outputs.
    """
    model1_outputs: Set[str] = {output.name for output in model1.graph.output}
    model2_outputs: Set[str] = {output.name for output in model2.graph.output}

    if model1_outputs != model2_outputs:
        if strict is True:
            raise ValueError(
                f"The two model protos outputs are expected to have the same number of outputs and output names "
                f"when strict=True. Found the outputs {model1_outputs - model2_outputs} only in model1, and "
                f"{model2_outputs - model1_outputs} only in model2."
            )
        else:
            logger.info(
                "The two models proto have different outputs (%d and %d "
                "outputs). Constant outputs will be added to unify the two models outputs.",
                len(model1_outputs),
                len(model2_outputs),
            )

    if model2_outputs.issubset(model1_outputs) is False:
        raise ValueError("The second ModelProto should not have more outputs than the first.")

    for idx in range(len(model1.graph.output)):
        model_output_1: str = model1.graph.output[idx]
        model_output_2: str = model2.graph.output[idx] if idx < len(model2.graph.output) else None

        if model_output_2 is None or model_output_1 != model_output_2:
            if model_output_2 is None or not (
                (model_output_1.name == model_output_2.name)
                and (model_output_1.type.tensor_type.elem_type == model_output_2.type.tensor_type.elem_type)
            ):
                if strict is False and model_output_1.name not in model2_outputs:
                    data_type = model_output_1.type.tensor_type.elem_type
                    dims_output_1: List[Any] = _infer_output_shape(model_output_1)
                    if not isinstance(dims_output_1[0], str):
                        raise ValueError(
                            f"Expected a dynamic shape for the axis zero of {model_output_1.name}, "
                            f"found a static shape: {dims_output_1[0]}"
                        )

                    # fill the constant shape with the original shape, except for the axis zero that is 0 for an
                    # empty constant, and the dynamic axis set to 1
                    dims_dummy_output = [0]
                    for dim in dims_output_1[1:]:
                        if isinstance(dim, str):
                            dims_dummy_output.append(1)
                        else:
                            dims_dummy_output.append(dim)

                    logger.info(
                        "Adding a constant output for %s of shape %s in model2.", model_output_1.name, dims_dummy_output
                    )
                    value: TensorProto = onnx.helper.make_tensor(
                        name="const_tensor", data_type=data_type, dims=dims_dummy_output, vals=[]
                    )
                    constant_node: NodeProto = onnx.helper.make_node(
                        "Constant",
                        name=f"Constant_{len(model2.graph.node) + 1}",
                        inputs=[],
                        outputs=[model_output_1.name],
                        value=value,
                    )
                    model2.graph.node.append(constant_node)

                    constant_empty_output: ValueInfoProto = onnx.helper.make_tensor_value_info(
                        model_output_1.name,
                        model_output_1.type.tensor_type.elem_type,
                        _infer_output_shape(model_output_1),
                    )
                    model2.graph.output.insert(idx, constant_empty_output)
                else:
                    if model_output_2 is not None:
                        raise ValueError(
                            f"Cannot match {model_output_1.name} with {model_output_2.name}. Make sure your"
                            f" model protos have same outputs, have same data types and are in the same order."
                        )
                    else:
                        raise ValueError(
                            f"Too few outputs of model2 were found to match with {model_output_1.name}."
                            f" Please try to pass strict=False, or fill a bug report at "
                            "https://github.com/huggingface/optimum."
                        )
            else:
                model2.graph.output.remove(model_output_2)

                # We use model1 (normally the decoder) for the output shape
                # TODO(shaahji): relax this, and keep the most permissive output shape between model1 and model2
                # while checking they are compatible
                new_output: ValueInfoProto = onnx.helper.make_tensor_value_info(
                    model_output_1.name,
                    model_output_1.type.tensor_type.elem_type,
                    _infer_output_shape(model_output_1),
                )
                model2.graph.output.insert(idx, new_output)

    if not all(
        model_output_1 == model_output_2
        for model_output_1, model_output_2 in zip(model1.graph.output, model2.graph.output)
    ):
        raise RuntimeError("Failed to unify outputs of given ONNX model protos.")


def _create_name_sharing_dict(
    duplicate_weights: DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]], suffix: str = ""
) -> Dict[Tuple[str, int], str]:
    """Create a map mapping old initializer names to new initializer names.

    As different ONNX models may use the same initializer name but need to be mapped to a different new name,
    the map is actually from (old name, model id) to new name.

    Example of initializers with the same name that will need to be mapped to a different one:
    Model 1 with:
    /transformer/Constant_8_output_0 of datatype 1

    Model 2 with:
    /transformer/Constant_8_output_0 of datatype 7

    Args:
        duplicate_weights (`DefaultDict[Tuple[int, bytes]`):
            Map of duplicate weights

        suffix (`str`, defaults to `""`):
            Suffix to append to common name

    """
    name_sharing_dict: Dict[Tuple[str, int], str] = {}
    used_common_names: Dict[str, int] = {}
    for duplicates in duplicate_weights.values():
        common_name, model_id = duplicates.pop()

        # this is needed in case two different groups of shared initializers may share the same name, for example
        # onnx::MatMul_2295 in the first model, and onnx::MatMul_2295 in the second model, although point to
        # different data
        used_common_names[common_name] = used_common_names.get(common_name, 0) + 1

        duplicates.add((common_name, model_id))
        for k in duplicates:
            assert k not in name_sharing_dict
            name_sharing_dict[k] = (
                f"{common_name}_{suffix}_{used_common_names[common_name]}" if suffix != "" else common_name
            )

    return name_sharing_dict


def _get_all_inputs(models: List[ModelProto]) -> List[ValueInfoProto]:
    """Collect unique inputs from all `models` into a single list."""
    inputs: List[ValueInfoProto] = []
    input_names: Set[str] = set()
    for model in models:
        for inp in model.graph.input:
            if inp.name not in input_names:
                input_names.add(inp.name)
                inputs.append(inp)

    return inputs


def _find_duplicate_initializers(
    models: List[ModelProto],
) -> DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]]:
    """Create a map (unique data) --> set of (initializer name, model id).

    Initializers with a dimension 0, or dimension 1 with data type int32 or int64, are not included in the
    generated map.
    """
    duplicates: DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]] = defaultdict(set)
    for i in range(len(models)):
        for initializer in models[i].graph.initializer:
            tensor_dims = tuple(initializer.dims)
            if (len(tensor_dims) > 1) or (
                (len(tensor_dims) == 1) and (initializer.data_type not in [TensorProto.INT32, TensorProto.INT64])
            ):
                # Extract tensor data as numpy array
                tensor_data: NDArray = onnx.numpy_helper.to_array(initializer)

                # Hash tensor data to avoid storing large amounts of data in memory
                hashed = hashlib.sha512()
                hashed.update(tensor_data)
                tensor_digest = hashed.hexdigest()

                duplicates[(initializer.data_type, tensor_digest, tensor_dims)].add((initializer.name, i))

    return duplicates


def _replace_input_names(models: List[ModelProto], name_sharing_dict: Dict[Tuple[str, int], str]):
    """Replace the names of node inputs from the models by the names in the name_sharing_dict."""
    for i in range(len(models)):
        for node in models[i].graph.node:
            for j in range(len(node.input)):
                name_id_pair = (node.input[j], i)
                if name_id_pair in name_sharing_dict:
                    node.input[j] = name_sharing_dict[name_id_pair]


def _deduplicated_cross_model_initializers(models: List[ModelProto], suffix: str = None) -> List[TensorProto]:
    """TODO(shaahji): short documentation."""
    duplicates: DefaultDict[Tuple[int, str, Tuple], Set[Tuple[str, int]]] = _find_duplicate_initializers(models)
    name_sharing_dict: Dict[Tuple[str, int], str] = _create_name_sharing_dict(duplicates, suffix=suffix)

    _replace_input_names(models, name_sharing_dict)

    deduplicated_initializers: List[TensorProto] = []
    deduplicated_name: Set[TensorProto] = set()

    for i in range(len(models)):
        for initializer in models[i].graph.initializer:
            name_id_pair = (initializer.name, i)
            if (name_id_pair in name_sharing_dict) and (name_sharing_dict[name_id_pair] not in deduplicated_name):
                deduplicated_name.add(name_sharing_dict[name_id_pair])
                initializer.name = name_sharing_dict[name_id_pair]
                deduplicated_initializers.append(initializer)

    return deduplicated_initializers


def _convert_graph_to_subgraph(graph: GraphProto, subgraph_name: str) -> GraphProto:
    # Keep initializers of dim 0 (or dim 1 + int32/int64) in subgraphs for readability purposes, and also because
    # ONNX Runtime breaks after optimization + merge if they are not
    initializers: List[TensorProto] = [
        initializer
        for initializer in graph.initializer
        if (len(initializer.dims) == 0)
        or ((len(initializer.dims) == 1) and (initializer.data_type in [TensorProto.INT32, TensorProto.INT64]))
    ]

    subgraph: GraphProto = onnx.helper.make_graph(
        nodes=graph.node,
        name=subgraph_name,
        inputs=[],
        outputs=graph.output,
        initializer=initializers,
    )

    return subgraph


def _merge_graphs_with_if_operator(
    all_inputs: List[ValueInfoProto],
    initializers: List[TensorProto],
    no_past_branch: GraphProto,
    with_past_branch: GraphProto,
    graph_name: str,
) -> GraphProto:
    # Merge subgraphs with a `If` node
    use_cache_branch: ValueInfoProto = onnx.helper.make_tensor_value_info(
        name="use_cache_branch",
        elem_type=onnx.TensorProto.BOOL,
        shape=[1],
    )
    if_node: NodeProto = onnx.helper.make_node(
        "If",
        inputs=["use_cache_branch"],
        outputs=[output.name for output in no_past_branch.output],
        name="olive::if",
        then_branch=with_past_branch,
        else_branch=no_past_branch,
    )
    merged_graph: GraphProto = onnx.helper.make_graph(
        nodes=[if_node],
        name=graph_name,
        inputs=[*all_inputs, use_cache_branch],
        outputs=no_past_branch.output,
        initializer=initializers,
    )
    return merged_graph


def merge_decoders(
    decoder: Union[ModelProto, Path, str],
    decoder_with_past: Union[ModelProto, Path, str],
    save_path: Optional[Union[Path, str]] = None,
    graph_name: str = "merged",
    strict: bool = True,
    save_as_external_data: bool = True,
    all_tensors_to_one_file: bool = True,
    check_model: bool = True,
) -> ModelProto:
    """Input decoder ONNX model and decoder with past ONNX model are merged into one ONNX model with if logic.

    Args:
        decoder (`Union[Path, str]`):
            Decoder ONNX model.
        decoder_with_past (`Union[Path, str]`):
            Decoder with past ONNX model.
        save_path (`Union[str, Path]`):
            The path to save merged ONNX model.
        graph_name (`str`, defaults to `"merged"`):
            Name of the parent graph (graph of the control flow node).
        strict (`bool`, defaults to `True`):
            When set, the decoder and decoder_with_past are expected to have strictly the same number of outputs.
            When False, the decoder is allowed to have more outputs that decoder_with_past, in which case constant
            outputs are added to match the number of outputs.
        save_as_external_data (`bool`):
            Serializes tensor data to separate files instead of directly in the ONNX file. Large models (>2GB)
            may be forced to save external data regardless of the value of this parameter.
        all_tensors_to_one_file (`bool`):
            Effective only if save_as_external_data is True. If true, save all tensors to one external file
            specified by 'external_data_name'. If false, save each tensor to a file named with the tensor name.
        check_model (`bool`):
            Check model after merging.

    """
    if isinstance(decoder, (str, Path)):
        decoder = onnx.load(Path(decoder).as_posix())

    if isinstance(decoder_with_past, (str, Path)):
        decoder_with_past = onnx.load(Path(decoder_with_past).as_posix())

    decoder_opset: int = _get_onnx_opset(decoder)
    decoder_with_past_opset: int = _get_onnx_opset(decoder_with_past)
    if decoder_opset != decoder_with_past_opset:
        raise ValueError(
            f"Decoder's opset is {decoder_opset}, but decoder with past's opset is {decoder_with_past_opset}. "
            "Make sure having the same opset before merging."
        )

    all_inputs: List[ValueInfoProto] = _get_all_inputs([decoder, decoder_with_past])

    # Replace the axis name `sequence_length` of the attention_mask input by `attention_mask_sequence_length`.
    # This is because the merged model `input_ids` and `attention_mask` inputs may not always have the same length
    # on the 2nd axis. In the first pass, `input_ids` and `attention_mask` are indeed of the same length, but in
    # later pass `input_ids` is of length 1 while `attention_mask` is of length `past_sequence_length + 1`
    for inp in all_inputs:
        if inp.name == "attention_mask":
            if inp.type.tensor_type.shape.dim[1].dim_param != "sequence_length":
                raise ValueError("Expected attention_mask second axis to be dynamic and named `sequence_length`.")

            inp.type.tensor_type.shape.dim[1].dim_param = "attention_mask_sequence_length"

    _unify_onnx_outputs(decoder, decoder_with_past, strict=strict)
    deduplicated_initializers: List[TensorProto] = _deduplicated_cross_model_initializers(
        [decoder, decoder_with_past], suffix=graph_name
    )

    # Make subgraphs
    no_past_branch: GraphProto = _convert_graph_to_subgraph(decoder.graph, "no_past")
    with_past_branch: GraphProto = _convert_graph_to_subgraph(decoder_with_past.graph, "with_past")

    # Merge subgraphs
    merged_graph: GraphProto = _merge_graphs_with_if_operator(
        all_inputs, deduplicated_initializers, no_past_branch, with_past_branch, graph_name
    )

    # Preserve imports from the decoder without/with past ONNX
    opset_imports: List[OperatorSetIdProto] = []
    opset_domains: Set[str] = set()
    for opset_import in list(decoder.opset_import) + list(decoder_with_past.opset_import):
        if opset_import.domain not in opset_domains:
            opset_imports.append(opset_import)
            opset_domains.add(opset_import.domain)

    merged_model: ModelProto = onnx.helper.make_model(merged_graph, opset_imports=opset_imports)

    if save_path:
        save_path = Path(save_path).as_posix()
        onnx.save(
            merged_model,
            save_path,
            save_as_external_data=save_as_external_data,
            all_tensors_to_one_file=all_tensors_to_one_file,
            location=os.path.basename(save_path) + ".data",
        )

        if check_model:
            try:
                onnx.checker.check_model(save_path)
            except Exception as e:
                if "No Op registered for" in str(e):
                    pass
                else:
                    raise

    return merged_model
