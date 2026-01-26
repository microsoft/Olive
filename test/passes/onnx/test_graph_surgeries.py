# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession

from olive.constants import MSFT_DOMAIN, OpType
from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.graph_surgeries import GraphSurgeries
from olive.passes.onnx.model_builder import ModelBuilder
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pytorch.rtn import Rtn
from test.utils import get_tiny_phi3, make_local_tiny_llama


def get_onnx_model(model_path):
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1])

    helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, [None])

    output = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [None])

    node1 = helper.make_node("Add", ["input1", "input2"], ["intermediate"], name="Add")
    node2 = helper.make_node("Relu", ["intermediate"], ["output1"], name="Relu")

    graph_def = helper.make_graph(
        [node1, node2],
        "test-model",
        [input1, input2],
        [output],
    )

    model = helper.make_model(graph_def, producer_name="onnx-example")
    onnx.save(model, model_path)
    return ONNXModelHandler(model_path=str(model_path))


def get_quantized_model(model_path):
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1])

    scale_initializer = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name="scale")
    zero_point_initializer = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="zero_point")

    helper.make_tensor_value_info("quantized_output", TensorProto.UINT8, [1])
    quantize_node = helper.make_node(
        "QuantizeLinear", inputs=["input1", "scale", "zero_point"], outputs=["quantized_output"], name="QuantizeNode"
    )
    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["quantized_output", "scale", "zero_point"],
        outputs=["output1"],
        name="DequantizeNode",
    )

    graph_def = helper.make_graph(
        [quantize_node, dequantize_node],
        "quantized-test-model",
        [input1],
        [output],
        initializer=[scale_initializer, zero_point_initializer],
    )

    model = helper.make_model(graph_def, producer_name="onnx-example")
    onnx.save(model, model_path)
    return ONNXModelHandler(model_path=str(model_path))


def test_rename_inputs(tmp_path):
    # setup
    renamed_input = "renamed_input"
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameInputs", "old_names": ["input1"], "new_names": [renamed_input]}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert renamed_input in [graph_input.name for graph_input in model_def.graph.input]


def test_rename_outputs(tmp_path):
    # setup
    renamed_output = "renamed_output"
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameOutputs", "old_names": ["output1"], "new_names": [renamed_output]}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert renamed_output in [output.name for output in model_def.graph.output]


def test_remove_shapes(tmp_path):
    # setup
    model_path = tmp_path / "model.onnx"
    input_model = get_onnx_model(model_path)
    output_folder = str(tmp_path / "onnx")
    input_model = onnx.shape_inference.infer_shapes(input_model.load_model())
    assert len(input_model.graph.value_info) > 0
    onnx.save(input_model, model_path)
    p = create_pass_from_dict(GraphSurgeries, {"surgeries": [{"surgeon": "RemoveShapes"}]}, disable_search=True)

    # execute
    onnx_model = p.run(ONNXModelHandler(model_path=str(model_path)), output_folder)

    # assert
    assert len(onnx_model.load_model().graph.value_info) == 0


def test_infer_shapes(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(GraphSurgeries, {"surgeries": [{"surgeon": "InferShapes"}]}, disable_search=True)

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    for output in onnx_model.load_model().graph.output:
        if output.name == "output1":
            tensor_type = output.type.tensor_type
            shape = tensor_type.shape
            dims = [dim.dim_value for dim in shape.dim]
            assert dims == [1]


def test_remove_initializer_from_inputs(tmp_path):
    # setup
    model_path = tmp_path / "model.onnx"
    input_tensor = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3])
    initializer_tensor = helper.make_tensor(
        name="const1",
        data_type=TensorProto.FLOAT,
        dims=[1, 3],
        vals=[1.0, 2.0, 3.0],
    )
    node = helper.make_node(
        "Add",
        inputs=["input1", "const1"],
        outputs=["output1"],
    )
    output_tensor = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        nodes=[node],
        name="TestGraph",
        inputs=[input_tensor, helper.make_tensor_value_info("const1", TensorProto.FLOAT, [1, 3])],
        outputs=[output_tensor],
        initializer=[initializer_tensor],
    )
    model = helper.make_model(graph)
    onnx.save(model, model_path)

    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "RemoveInitializerFromInputs"}]}, disable_search=True
    )

    # execute
    output_model = p.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "onnx"))

    # assert
    assert Path(output_model.model_path).exists()
    output_model = output_model.load_model()
    assert "const1" not in {graph_input.name for graph_input in output_model.graph.input}


def test_reorder_inputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "ReorderInputs", "permutation": [1, 0]}]}, disable_search=True
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert [graph_input.name for graph_input in model_def.graph.input] == ["input2", "input1"]


def test_replace_erf_with_tanh(tmp_path):
    # setup
    model_path = tmp_path / "model.onnx"
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("erf_output", TensorProto.FLOAT, [1, 3])
    erf_node = helper.make_node("Erf", inputs=["input"], outputs=["erf_output"], name="ErfNode")
    graph_def = helper.make_graph(nodes=[erf_node], name="ErfTestGraph", inputs=[input_tensor], outputs=[output_tensor])
    model = helper.make_model(graph_def, producer_name="onnx-example")
    onnx.save(model, model_path)
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ReplaceErfWithTanh"}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "onnx"))

    # assert
    model_def = onnx_model.load_model()
    tanh_node = next(node for node in model_def.graph.node if node.op_type == "Tanh")
    mul_node = next(node for node in model_def.graph.node if node.op_type == "Mul")

    scale_initializer = next(init for init in model_def.graph.initializer if init.name == mul_node.input[1])
    scale_value = np.array(scale_initializer.float_data, dtype=np.float32)
    assert np.isclose(scale_value, 605 / 503, atol=1e-6), "Scale value mismatch"
    assert tanh_node.input[0] == mul_node.output[0], "Tanh input should match Mul output"
    assert tanh_node.output[0] == "erf_output", "Tanh output should replace Erf output"


def test_zero_out_input(tmp_path):
    # setup
    input_model_path = tmp_path / "model.onnx"
    input_model = get_onnx_model(input_model_path)
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ZeroOutInput", "node_name": "Add", "input_idx": 1}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    add_node = next(node for node in model_def.graph.node if node.name == "Add")
    zero_node = next(node for node in model_def.graph.node if node.name == "Add_zero")
    assert add_node.input[1] == zero_node.output[0]

    zero_tensor = zero_node.attribute[0].t
    zero_values = np.array(zero_tensor.float_data, dtype=np.float32).reshape(zero_tensor.dims)
    assert np.all(zero_values == 0), "Zero tensor should contain all zeros."


def test_remove_inputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "RemoveInputs", "names": ["input1"]}]}, disable_search=True
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert [graph_input.name for graph_input in model_def.graph.input] == ["input2"]
    for node in model_def.graph.node:
        assert "input1" not in node.input


def test_expose_outputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "ExposeOutputs", "names": ["Add"]}]}, disable_search=True
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert [output.name for output in model_def.graph.output] == ["output1", "intermediate"]


def test_expose_quantized_output(tmp_path):
    # setup
    input_model_path = tmp_path / "quantized_model.onnx"
    input_model = get_quantized_model(input_model_path)
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ExposeQuantizedOutput", "output_name": "output1"}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    output_model = onnx_model.load_model()

    # Retrieve the quantized output name from the original model
    dequantized_node = next(node for node in input_model.load_model().graph.node if node.op_type == "DequantizeLinear")
    quantized_output_name = dequantized_node.input[0]

    # Retrieve the scale and zero_point initializer names from the original model
    quantized_node = next(node for node in input_model.load_model().graph.node if quantized_output_name in node.output)
    scale_name_in_original_model = quantized_node.input[1]
    zero_point_name_in_original_model = quantized_node.input[2]

    # Retrieve the scale and zero_point values from the original model
    original_scale_initializer = next(
        init for init in input_model.load_model().graph.initializer if init.name == scale_name_in_original_model
    )
    original_scale_value = numpy_helper.to_array(original_scale_initializer)[0]

    original_zero_point_initializer = next(
        init for init in input_model.load_model().graph.initializer if init.name == zero_point_name_in_original_model
    )
    original_zero_point_value = numpy_helper.to_array(original_zero_point_initializer)[0]
    zero_point_dtype = helper.tensor_dtype_to_np_dtype(original_zero_point_initializer.data_type)

    # Validate that the quantized output is exposed in the modified model
    exposed_outputs = [output.name for output in output_model.graph.output]
    assert quantized_output_name in exposed_outputs, "Quantized output not exposed."
    assert "output1" not in exposed_outputs, "Original output should be removed."

    # Construct expected node names for scale and zero_point
    output_name = "output1"
    scale_node_name = f"{output_name}_exposed_scale"
    zero_point_node_name = f"{output_name}_exposed_zero_point"

    # Validate that the scale node and its initializer exist in the modified model
    scale_node = next((node for node in output_model.graph.node if node.name == scale_node_name), None)
    assert scale_node is not None, "Scale node not added."
    # After deduplication, the node may reference the original initializer, so get the actual input name
    scale_initializer_name = scale_node.input[0]
    scale_initializer = next(init for init in output_model.graph.initializer if init.name == scale_initializer_name)
    assert np.allclose(numpy_helper.to_array(scale_initializer), np.array([original_scale_value], dtype=np.float32)), (
        "Scale value mismatch."
    )

    # Validate that the zero_point node and its initializer exist in the modified model
    zero_point_node = next((node for node in output_model.graph.node if node.name == zero_point_node_name), None)
    assert zero_point_node is not None, "Zero point node not added."
    # After deduplication, the node may reference the original initializer, so get the actual input name
    zero_point_initializer_name = zero_point_node.input[0]
    zero_point_initializer = next(
        init for init in output_model.graph.initializer if init.name == zero_point_initializer_name
    )
    assert np.allclose(
        numpy_helper.to_array(zero_point_initializer), np.array([original_zero_point_value], dtype=zero_point_dtype)
    ), "Zero point value mismatch."


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, use_rsqrt=True, use_cast=True, all_ones=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size) if all_ones else torch.randn(hidden_size))
        self.variance_epsilon = eps
        self.use_rsqrt = use_rsqrt
        self.use_cast = use_cast

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        if self.use_cast:
            hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        if self.use_rsqrt:
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * (hidden_states.to(input_dtype) if self.use_cast else hidden_states)


def check_l2norm(
    original_model_path: str,
    modified_model_path: str,
    hidden_size: int,
    expected_num_nodes: int,
    check_all_ones: int,
    has_skip: bool = False,
):
    # check output values match
    input_session = InferenceSession(original_model_path)
    output_session = InferenceSession(modified_model_path)
    input_feed = {"x": np.random.randn(1, hidden_size).astype(np.float32)}
    if has_skip:
        input_feed["skip"] = np.random.randn(1, hidden_size).astype(np.float32)
    input_result = input_session.run(None, input_feed)
    output_result = output_session.run(None, input_feed)
    for i_r, o_r in zip(input_result, output_result):
        np.testing.assert_allclose(i_r, o_r, rtol=1e-3, atol=1e-3)

    # count nodes
    dag = OnnxDAG.from_model_path(modified_model_path)
    assert len(dag.nodes) == expected_num_nodes
    assert "LpNormalization" in dag.get_node_op_types()

    # check all ones case
    if check_all_ones:
        mul_name = None
        for node in dag.get_node_names():
            if dag.get_node_op_type(node) == "Mul":
                mul_name = node
                break
        mul_weight_name = None
        for input_name in dag.get_node_inputs(mul_name):
            if dag.is_initializer(input_name):
                mul_weight_name = input_name
                break
        mul_weight = dag.get_initializer_np_array(mul_weight_name)
        assert mul_weight.shape == (1,)
        assert np.allclose(mul_weight, np.sqrt(hidden_size))


@pytest.mark.parametrize("use_rsqrt", [True, False])
@pytest.mark.parametrize("use_cast", [True, False])
@pytest.mark.parametrize("all_ones", [True, False])
def test_rmsnorm_to_l2norm(tmp_path, use_rsqrt, use_cast, all_ones):
    # setup
    hidden_size = 3
    module = RMSNorm(hidden_size, use_rsqrt=use_rsqrt, use_cast=use_cast, all_ones=all_ones)
    input_model_path = tmp_path / "input_model.onnx"
    # Use TorchScript export because the RMSNormToL2Norm surgery pattern relies on
    # TorchScript-specific graph structure which differs from dynamo export
    torch.onnx.export(
        module,
        torch.randn(1, hidden_size),
        input_model_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=20,
        dynamo=False,
    )
    input_model = ONNXModelHandler(input_model_path)

    output_folder = str(tmp_path / "output")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RMSNormToL2Norm"}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    check_l2norm(
        str(input_model_path),
        onnx_model.model_path,
        hidden_size,
        2 + 2 * int(use_cast),
        all_ones,
    )


@pytest.mark.parametrize("all_ones", [True, False])
def test_simplifiedlayernorm_to_l2norm(tmp_path, all_ones):
    # setup
    hidden_size = 3
    inputs = [
        onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, hidden_size]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, hidden_size]),
    ]
    weight = (np.ones(hidden_size) if all_ones else np.random.randn(hidden_size)).astype(np.float32)
    initializers = [onnx.numpy_helper.from_array(weight, name="weight")]
    nodes = [
        onnx.helper.make_node(
            "SimplifiedLayerNormalization",
            inputs=["x", "weight"],
            outputs=["layernorm_output"],
            name="layernorm/LayerNorm",
        ),
        onnx.helper.make_node("Identity", inputs=["layernorm_output"], outputs=["y"], name="Identity"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.save(model, str(tmp_path / "input_model.onnx"))
    input_model = ONNXModelHandler(model_path=str(tmp_path / "input_model.onnx"))

    output_folder = str(tmp_path / "output")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "SimplifiedLayerNormToL2Norm"}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    check_l2norm(str(tmp_path / "input_model.onnx"), onnx_model.model_path, hidden_size, 3, all_ones)


@pytest.mark.parametrize("all_ones", [True, False])
@pytest.mark.parametrize("output_skip_sum", [True, False])
def test_simplifiedlayernorm_to_l2norm_skip(tmp_path, all_ones, output_skip_sum):
    # setup
    hidden_size = 3
    inputs = [
        onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, hidden_size]),
        onnx.helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, hidden_size]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, hidden_size]),
    ]
    if output_skip_sum:
        outputs.append(
            onnx.helper.make_tensor_value_info("skip_sum", TensorProto.FLOAT, [1, hidden_size]),
        )
    initializers = [
        onnx.numpy_helper.from_array(
            (np.ones(hidden_size) if all_ones else np.random.randn(hidden_size)).astype(np.float32), name="weight"
        )
    ]
    nodes = [
        onnx.helper.make_node(
            "SkipSimplifiedLayerNormalization",
            inputs=["x", "skip", "weight"],
            outputs=["layernorm_output"] if not output_skip_sum else ["layernorm_output", "", "", "layernorm_skip_sum"],
            name="layernorm/LayerNorm",
            domain=MSFT_DOMAIN,
        ),
        onnx.helper.make_node("Identity", inputs=["layernorm_output"], outputs=["y"], name="Identity"),
    ]
    if output_skip_sum:
        nodes.append(
            onnx.helper.make_node(
                "Identity", inputs=["layernorm_skip_sum"], outputs=["skip_sum"], name="Identity_skip_sum"
            )
        )
    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.save(model, str(tmp_path / "input_model.onnx"))
    input_model = ONNXModelHandler(model_path=str(tmp_path / "input_model.onnx"))

    output_folder = str(tmp_path / "output")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "SimplifiedLayerNormToL2Norm"}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    check_l2norm(
        str(tmp_path / "input_model.onnx"),
        output_model.model_path,
        hidden_size,
        4 + int(output_skip_sum),
        all_ones,
        has_skip=True,
    )


@pytest.mark.parametrize("use_large_cache", [True, False])
def test_remove_rope_multi_cache(tmp_path, use_large_cache):
    # setup
    tiny_model = get_tiny_phi3()
    local_tiny_path = tmp_path / "input_model"
    tiny_model.load_model().save_pretrained(local_tiny_path)
    tiny_model.save_metadata(local_tiny_path)
    config_json_path = local_tiny_path / "config.json"
    with config_json_path.open() as f:
        config = json.load(f)
    # change the max position embedding to 10000
    config["max_position_embeddings"] = 10000
    config["rope_scaling"] = {
        "long_factor": [1] * 12,
        "short_factor": [1] * 12,
        "type": "longrope",
    }
    del config["auto_map"]
    with config_json_path.open("w") as f:
        json.dump(config, f)

    input_model = create_pass_from_dict(
        ModelBuilder,
        {"precision": "fp32"},
        disable_search=True,
    ).run(HfModelHandler(local_tiny_path), str(tmp_path / "onnx"))

    output_folder = str(tmp_path / "output")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RemoveRopeMultiCache", "use_large_cache": use_large_cache}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    dag = OnnxDAG.from_model_path(output_model.model_path)
    assert "If" not in dag.get_node_op_types()
    assert dag.get_initializer_np_array("cos_cache_single").shape[0] == 10000 if use_large_cache else 4096


def test_attention_mask_to_sequence_lengths(tmp_path):
    # setup
    input_model = create_pass_from_dict(
        ModelBuilder,
        {"precision": "fp32"},
        disable_search=True,
    ).run(make_local_tiny_llama(tmp_path), str(tmp_path / "onnx"))

    output_folder = str(tmp_path / "output")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "AttentionMaskToSequenceLengths"}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    output_model_input_names = output_model.io_config["input_names"]
    assert "attention_mask" not in output_model_input_names
    assert "past_seq_len" in output_model_input_names
    assert "total_seq_len" in output_model_input_names


def test_replace_attention_mask_value(tmp_path):
    # setup
    min_value = float(np.finfo(np.float32).min)
    input_tensors = [
        helper.make_tensor_value_info("input1", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("input3", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("input4", TensorProto.FLOAT, [1]),
    ]
    output_tensors = [
        helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("output3", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("output4", TensorProto.FLOAT, [4, 1, 2, 2]),
        helper.make_tensor_value_info("output5", TensorProto.FLOAT, [1, 1, 2, 2]),
    ]
    expand_init = np.array([[[[0, min_value], [min_value, min_value]]]], dtype=np.float32)
    initializers = [
        helper.make_tensor("init", TensorProto.FLOAT, [], [min_value]),
        numpy_helper.from_array(expand_init, name="add_init"),
        helper.make_tensor("expand_shape", TensorProto.INT64, [4], [4, 1, 2, 2]),
    ]
    nodes = [
        helper.make_node(
            "ConstantOfShape",
            inputs=["input1"],
            outputs=["output1"],
            name="ConstantOfShape",
            value=helper.make_tensor("", TensorProto.FLOAT, [1], [min_value]),
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["Constant_output"],
            name="Constant",
            value=helper.make_tensor("", TensorProto.FLOAT, [], [min_value]),
        ),
        helper.make_node(
            "Mul",
            inputs=["input2", "Constant_output"],
            outputs=["output2"],
            name="Mul_constant",
        ),
        helper.make_node(
            "Mul",
            inputs=["input3", "init"],
            outputs=["output3"],
            name="Mul_init",
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["expand_constant"],
            name="Constant_expand",
            value=numpy_helper.from_array(expand_init, name=""),
        ),
        helper.make_node(
            "Expand",
            inputs=["expand_constant", "expand_shape"],
            outputs=["output4"],
            name="Expand_constant",
        ),
        helper.make_node(
            "Add",
            inputs=["input4", "add_init"],
            outputs=["output5"],
            name="Add_init",
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=input_tensors,
        outputs=output_tensors,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ReplaceAttentionMaskValue"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())
    output_session = output_model.prepare_session()
    outputs = output_session.run(
        None,
        {
            "input1": np.array([1], dtype=np.int64),
            "input2": np.array([1], dtype=np.float32),
            "input3": np.array([1], dtype=np.float32),
            "input4": np.array([0], dtype=np.float32),
        },
    )
    assert all(o == -1e4 for o in outputs[:3])
    expected_output_4 = np.repeat(expand_init, 4, axis=0)
    expected_output_4[expected_output_4 == min_value] = -1e4
    assert np.array_equal(outputs[3], expected_output_4)
    expected_output_5 = expand_init.copy()
    expected_output_5[expected_output_5 == min_value] = -1e4
    assert np.array_equal(outputs[4], expected_output_5)


def test_matmul_add_to_gemm(tmp_path):
    # setup input and output tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 3])

    constant1_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    constant2_data = np.array([1, 2, 3], dtype=np.float32)
    # Create constant tensors
    initializers = [
        helper.make_tensor("constant1", TensorProto.FLOAT, [3, 3], constant1_data),
        helper.make_tensor("constant2", TensorProto.FLOAT, [3], constant2_data),
    ]

    nodes = [
        helper.make_node("MatMul", inputs=["input", "constant1"], outputs=["matmul_output"]),
        helper.make_node("Add", inputs=["matmul_output", "constant2"], outputs=["inter"]),
        helper.make_node("Identity", inputs=["inter"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "MatMulAddToGemm"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # Matmul->Add->Identity will be replaced with Reshape->Gemm->Reshape->Identity
    expected_num_nodes = 4
    dag = OnnxDAG.from_model_path(output_model.model_path)
    assert len(dag.nodes) == expected_num_nodes
    assert "MatMul" not in dag.get_node_op_types()

    # assert
    onnx.checker.check_model(output_model.load_model())
    output_session = output_model.prepare_session()

    # Define the input data
    input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)

    outputs = output_session.run(None, {"input": input_data})

    matmul_output = np.matmul(input_data, constant1_data)
    expected_output = matmul_output + constant2_data
    assert np.allclose(outputs[0], expected_output)


def test_matmul_add_to_gemm_with_relu(tmp_path):
    # setup input and output tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 3])

    constant1_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    constant2_data = np.array([1, 2, 3], dtype=np.float32)
    # Create constant tensors
    initializers = [
        helper.make_tensor("constant1", TensorProto.FLOAT, [3, 3], constant1_data),
        helper.make_tensor("constant2", TensorProto.FLOAT, [3], constant2_data),
    ]

    nodes = [
        helper.make_node("MatMul", inputs=["input", "constant1"], outputs=["matmul_output"]),
        helper.make_node("Add", inputs=["matmul_output", "constant2"], outputs=["add_output"]),
        helper.make_node("Relu", inputs=["add_output"], outputs=["relu_output"]),
        helper.make_node("Identity", inputs=["relu_output"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "MatMulAddToGemm"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # Matmul->Add->Relu->Identity will be replaced with Reshape->Gemm->Relu->Reshape->Identity
    expected_num_nodes = 5
    dag = OnnxDAG.from_model_path(output_model.model_path)
    assert len(dag.nodes) == expected_num_nodes
    assert "MatMul" not in dag.get_node_op_types()
    gemm_node = None
    for node_name in dag.get_node_names():
        if dag.get_node_op_type(node_name) == "Gemm":
            gemm_node = node_name
            break
    assert dag.get_node_op_type(dag.get_consumers(gemm_node)[0]) == "Relu"

    # assert
    onnx.checker.check_model(output_model.load_model())
    output_session = output_model.prepare_session()

    # Define the input data
    input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)

    outputs = output_session.run(None, {"input": input_data})

    matmul_output = np.matmul(input_data, constant1_data)
    add_output = matmul_output + constant2_data
    expected_output = np.maximum(add_output, 0)
    assert np.allclose(outputs[0], expected_output)


@pytest.mark.parametrize("keep_clip_after_inputs", [True, False])
def test_remove_qdq(tmp_path, keep_clip_after_inputs):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    scale_initializer = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name="scale")
    zero_point_initializer = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="zero_point")

    nodes = [
        helper.make_node("QuantizeLinear", inputs=["input", "scale", "zero_point"], outputs=["quantized"]),
        helper.make_node("DequantizeLinear", inputs=["quantized", "scale", "zero_point"], outputs=["dequantized"]),
        helper.make_node("Relu", inputs=["dequantized"], outputs=["relu_output"]),
        helper.make_node("QuantizeLinear", inputs=["relu_output", "scale", "zero_point"], outputs=["quantized2"]),
        helper.make_node("DequantizeLinear", inputs=["quantized2", "scale", "zero_point"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[scale_initializer, zero_point_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RemoveQDQ", "keep_clip_after_inputs": keep_clip_after_inputs}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert - check the model was modified
    output_model_def = output_model.load_model()
    op_types = [node.op_type for node in output_model_def.graph.node]
    assert "QuantizeLinear" not in op_types
    assert "DequantizeLinear" not in op_types
    if keep_clip_after_inputs:
        assert "Clip" in op_types


def test_matmul_to_transpose_conv_transpose(tmp_path):
    # setup
    batch_size, seq_len, hidden_size = 2, 128, 768
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, 3072])

    weight_data = np.random.randn(hidden_size, 3072).astype(np.float32)
    weight_initializer = numpy_helper.from_array(weight_data, name="weight")

    nodes = [
        helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "MatMulToTransposeConvTranspose"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert - just check that the transform modified the graph
    dag = OnnxDAG.from_model_path(output_model.model_path)
    # The transform should have replaced MatMul with Conv
    assert "Conv" in dag.get_node_op_types()
    assert "MatMul" not in dag.get_node_op_types()


def test_remove_intermediary_squeeze_and_unsqueeze(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 224])

    # For ONNX opset 13+, axes should be an input, not an attribute
    squeeze_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="squeeze_axes")
    unsqueeze_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="unsqueeze_axes")

    nodes = [
        # Add an Identity node so Squeeze is not directly connected to input
        helper.make_node("Identity", inputs=["input"], outputs=["identity_output"]),
        helper.make_node("Squeeze", inputs=["identity_output", "squeeze_axes"], outputs=["squeezed"]),
        helper.make_node("Relu", inputs=["squeezed"], outputs=["relu_output"]),
        helper.make_node("Unsqueeze", inputs=["relu_output", "unsqueeze_axes"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[squeeze_axes, unsqueeze_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RemoveIntermediarySqueezeAndUnsqueeze"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    dag = OnnxDAG.from_model_path(output_model.model_path)
    assert "Squeeze" not in dag.get_node_op_types()
    assert "Unsqueeze" not in dag.get_node_op_types()
    assert "Relu" in dag.get_node_op_types()


def test_qdq_to_clip(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    scale_initializer = numpy_helper.from_array(np.array([0.01], dtype=np.float32), name="scale")
    zero_point_initializer = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="zero_point")

    nodes = [
        helper.make_node("QuantizeLinear", inputs=["input", "scale", "zero_point"], outputs=["quantized"]),
        helper.make_node("DequantizeLinear", inputs=["quantized", "scale", "zero_point"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[scale_initializer, zero_point_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "QDQToClip"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    dag = OnnxDAG.from_model_path(output_model.model_path)
    assert "Clip" in dag.get_node_op_types()
    assert "QuantizeLinear" not in dag.get_node_op_types()
    assert "DequantizeLinear" not in dag.get_node_op_types()


def test_remove_deqlin(tmp_path):
    # setup - RemoveDeqLin expects the DequantizeLinear output to feed into a Transpose
    input_tensor = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 224, 224, 3])

    scale_initializer = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name="scale")
    zero_point_initializer = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="zero_point")

    nodes = [
        helper.make_node("DequantizeLinear", inputs=["input", "scale", "zero_point"], outputs=["dequant_output"]),
        helper.make_node("Transpose", inputs=["dequant_output"], outputs=["output"], perm=[0, 2, 3, 1]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[scale_initializer, zero_point_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RemoveDeqLin"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert - RemoveDeqLin removes DequantizeLinear when followed by Transpose
    dag = OnnxDAG.from_model_path(output_model.model_path)
    # The transform removes the DequantizeLinear node
    assert "Transpose" in dag.get_node_op_types()


def test_non4d_model_inputs(tmp_path):
    # setup - create model with 3D input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 20, 30])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10, 20, 30])

    nodes = [
        helper.make_node("Relu", inputs=["input"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DModelInputs"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert - should have Unsqueeze nodes added to make inputs 4D
    model_def = output_model.load_model()
    op_types = [node.op_type for node in model_def.graph.node]
    assert "Unsqueeze" in op_types

    # The original input should still be 3D, but there should be an Unsqueeze node after it
    input_shape = model_def.graph.input[0].type.tensor_type.shape
    assert len(input_shape.dim) == 3

    # Check that the Unsqueeze node is connected to the input
    unsqueeze_nodes = [node for node in model_def.graph.node if node.op_type == "Unsqueeze"]
    assert len(unsqueeze_nodes) > 0
    assert any(node.input[0] == "input" for node in unsqueeze_nodes)


def test_non4d_model_outputs(tmp_path):
    # setup - create model with 3D output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 150528])  # flattened

    nodes = [
        helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DModelOutputs"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert - should have Reshape nodes added
    dag = OnnxDAG.from_model_path(output_model.model_path)
    op_types = dag.get_node_op_types()
    assert "Reshape" in op_types or len(dag.nodes) > 1


def test_standalone_reducesum(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])

    # For ONNX opset 13+, axes should be an input, not an attribute
    axes_tensor = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="axes")

    nodes = [
        helper.make_node("ReduceSum", inputs=["input", "axes"], outputs=["output"], keepdims=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[axes_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "StandaloneReduceSum"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_gather_transform(tmp_path):
    # setup
    data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, [10, 20])
    indices_tensor = helper.make_tensor_value_info("indices", TensorProto.INT64, [5])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 20])

    nodes = [
        helper.make_node("Gather", inputs=["data", "indices"], outputs=["output"], axis=0),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[data_tensor, indices_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Gather"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_gatherelements_transform(tmp_path):
    # setup
    data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4])
    indices_tensor = helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 2])

    nodes = [
        helper.make_node("GatherElements", inputs=["data", "indices"], outputs=["output"], axis=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[data_tensor, indices_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "GatherElements"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_non4d_initializers(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # 3D initializer
    bias_data = np.random.randn(3, 224, 224).astype(np.float32)
    bias_initializer = numpy_helper.from_array(bias_data, name="bias")

    nodes = [
        helper.make_node("Add", inputs=["input", "bias"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[bias_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DInitializers"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())
    dag = OnnxDAG.from_model_path(output_model.model_path)
    # check that bias is now 4D
    bias_init = dag.get_initializer_np_array("bias")
    assert len(bias_init.shape) == 4


def test_remove_all_tensor_value_shapes(tmp_path):
    # setup
    model_path = tmp_path / "model.onnx"
    input_model = get_onnx_model(model_path)
    # Add shape inference first
    input_model_with_shapes = onnx.shape_inference.infer_shapes(input_model.load_model())
    assert len(input_model_with_shapes.graph.value_info) > 0
    onnx.save(input_model_with_shapes, model_path)

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RemoveAllTensorValueShapes"}]},
        disable_search=True,
    )

    output_model = p.run(ONNXModelHandler(model_path=str(model_path)), output_folder)

    # assert - the transform removes shape information from value_info tensors
    model_def = output_model.load_model()
    # Check that shape information has been cleared
    for value_info in model_def.graph.value_info:
        tensor_type = value_info.type.tensor_type
        assert not tensor_type.HasField("shape") or len(tensor_type.shape.dim) == 0


def test_non4d_reshape(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 12])
    shape_tensor = numpy_helper.from_array(np.array([2, 12], dtype=np.int64), name="shape")

    nodes = [
        helper.make_node("Reshape", inputs=["input", "shape"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DReshape"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_non4d_expand(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])
    shape_tensor = numpy_helper.from_array(np.array([3, 4], dtype=np.int64), name="shape")

    nodes = [
        helper.make_node("Expand", inputs=["input", "shape"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DExpand"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_non4d_transpose(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 3, 2])

    nodes = [
        helper.make_node("Transpose", inputs=["input"], outputs=["output"], perm=[2, 1, 0]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DTranspose"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_non4d_slice(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 20, 30])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 10, 15])

    starts = numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([5, 10, 15], dtype=np.int64), name="ends")

    nodes = [
        helper.make_node("Slice", inputs=["input", "starts", "ends"], outputs=["output"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[starts, ends],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DSlice"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_non4d_lpnorm(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4])

    nodes = [
        helper.make_node("LpNormalization", inputs=["input"], outputs=["output"], axis=-1, p=2),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Non4DLpNorm"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_flatten_transform(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 5])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 60])

    nodes = [
        helper.make_node("Flatten", inputs=["input"], outputs=["output"], axis=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "Flatten"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())
    dag = OnnxDAG.from_model_path(output_model.model_path)
    # Flatten might be replaced with Reshape
    assert "Flatten" in dag.get_node_op_types() or "Reshape" in dag.get_node_op_types()


def test_add_intermediate_tensors_to_outputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "AddIntermediateTensorsToOutputs", "intermediate_tensor_to_add": ["intermediate"]}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    model_def = output_model.load_model()
    output_names = [output.name for output in model_def.graph.output]
    assert "intermediate" in output_names
    assert "output1" in output_names


def test_reshape_reducesum(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [6, 1])

    shape_tensor = numpy_helper.from_array(np.array([6, 4], dtype=np.int64), name="shape")

    # For ONNX opset 13+, axes should be an input, not an attribute
    reduce_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="reduce_axes")

    nodes = [
        helper.make_node("Reshape", inputs=["input", "shape"], outputs=["reshaped"]),
        helper.make_node("ReduceSum", inputs=["reshaped", "reduce_axes"], outputs=["output"], keepdims=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_tensor, reduce_axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ReshapeReduceSum"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_reshape_clip_reducesum(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [6, 1])

    shape_tensor = numpy_helper.from_array(np.array([6, 4], dtype=np.int64), name="shape")

    # For ONNX opset 13+, axes should be an input for ReduceSum, and min/max for Clip
    reduce_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="reduce_axes")
    clip_min = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="clip_min")
    clip_max = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name="clip_max")

    nodes = [
        helper.make_node("Reshape", inputs=["input", "shape"], outputs=["reshaped"]),
        helper.make_node("Clip", inputs=["reshaped", "clip_min", "clip_max"], outputs=["clipped"]),
        helper.make_node("ReduceSum", inputs=["clipped", "reduce_axes"], outputs=["output"], keepdims=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_tensor, reduce_axes, clip_min, clip_max],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ReshapeClipReduceSum"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_reducemax_transform(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 5])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 1, 1])

    nodes = [
        helper.make_node("ReduceMax", inputs=["input"], outputs=["output"], axes=[2, 3], keepdims=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ReduceMax"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    onnx.checker.check_model(output_model.load_model())


def test_quickgelu_to_sigmoid(tmp_path):
    # setup
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    nodes = [
        helper.make_node(
            OpType.QuickGelu, inputs=["input"], outputs=["output"], name="quickgelu_node", domain=MSFT_DOMAIN
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid(MSFT_DOMAIN, 1)])
    model.ir_version = 10

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "DecomposeQuickGelu"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # assert
    ir_model = output_model.load_ir_model()

    # Check that QuickGelu is replaced with Mul->Sigmoid->Mul
    op_types = [node.op_type for node in ir_model.graph.all_nodes()]
    assert OpType.QuickGelu not in op_types
    assert OpType.Mul in op_types
    assert OpType.Sigmoid in op_types
    assert op_types.count(OpType.Mul) == 2


def test_decompose_rotary_embedding(tmp_path):
    # setup
    batch_size, seq_len, hidden_size = 2, 8, 64
    max_seq_len = 128
    rotary_dim = hidden_size // 2

    # Generate position embeddings for cos/sin caches
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, rotary_dim, 2) * -(np.log(10000.0) / rotary_dim))

    cos_cache = np.zeros((max_seq_len, rotary_dim), dtype=np.float32)
    sin_cache = np.zeros((max_seq_len, rotary_dim), dtype=np.float32)
    cos_cache[:, 0::2] = np.cos(position * div_term)
    cos_cache[:, 1::2] = np.cos(position * div_term)
    sin_cache[:, 0::2] = np.sin(position * div_term)
    sin_cache[:, 1::2] = np.sin(position * div_term)

    cos_initializer = numpy_helper.from_array(cos_cache, name="cos_cache")
    sin_initializer = numpy_helper.from_array(sin_cache, name="sin_cache")

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size])
    position_ids_tensor = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [batch_size, seq_len])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size])

    nodes = [
        helper.make_node(
            OpType.RotaryEmbedding,
            inputs=["input", "position_ids", "cos_cache", "sin_cache"],
            outputs=["output"],
            name="rotary_embedding_node",
            domain=MSFT_DOMAIN,
            interleaved=0,
            num_heads=0,
            rotary_embedding_dim=hidden_size,
            scale=1.0,
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[input_tensor, position_ids_tensor],
        outputs=[output_tensor],
        initializer=[cos_initializer, sin_initializer],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(MSFT_DOMAIN, 1)])
    model.ir_version = 10

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    input_model = ONNXModelHandler(model_path=str(model_path))

    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "DecomposeRotaryEmbedding"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)

    # Check that the RotaryEmbedding node was replaced with expected nodes
    transformed_model = onnx.load(output_model.model_path)
    node_types = {node.op_type for node in transformed_model.graph.node}

    assert OpType.RotaryEmbedding not in node_types

    expected_nodes = {
        OpType.Reshape,
        OpType.Gather,
        OpType.Shape,
        OpType.Constant,
        OpType.Slice,
        OpType.Mul,
        OpType.Sub,
        OpType.Add,
        OpType.Concat,
        OpType.Div,
    }

    for node_type in expected_nodes:
        assert node_type in node_types, f"Expected node type {node_type} not found in transformed model"

    # Prepare test inputs
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    position_ids = np.arange(seq_len)[np.newaxis, :].repeat(batch_size, axis=0).astype(np.int64)

    # Compute expected output (reference implementation)
    position_ids_flat = position_ids.flatten()
    cos_gathered = cos_cache[position_ids_flat].reshape(batch_size, seq_len, rotary_dim)
    sin_gathered = sin_cache[position_ids_flat].reshape(batch_size, seq_len, rotary_dim)

    x_real = input_data[:, :, :rotary_dim]
    x_imag = input_data[:, :, rotary_dim:]

    output_real = x_real * cos_gathered - x_imag * sin_gathered
    output_imag = x_real * sin_gathered + x_imag * cos_gathered
    expected_output = np.concatenate([output_real, output_imag], axis=-1)

    # Run inference on lowered model
    sess = InferenceSession(output_model.model_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"input": input_data, "position_ids": position_ids})
    actual_output = outputs[0]

    # Verify outputs match
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)


@patch("olive.passes.onnx.graph_surgeries.DeduplicateHashedInitializersPass")
def test_deduplicate_hashed_initializers_pass_called(mock_dedup_pass, tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")

    # Create a pass with a simple surgery
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameInputs", "old_names": ["input1"], "new_names": ["renamed_input"]}]},
        disable_search=True,
    )

    ir_model = input_model.load_ir_model()
    mock_result = MagicMock()
    mock_result.model = ir_model
    mock_instance = MagicMock(return_value=mock_result)
    mock_dedup_pass.return_value = mock_instance

    # execute
    p.run(input_model, output_folder)

    # assert
    mock_dedup_pass.assert_called_once()
    mock_instance.assert_called_once()


# Skip: Both dynamo (TorchExportError) and TorchScript (RuntimeError: unordered_map::at)
# fail to export this model due to transformers/PyTorch version incompatibility
@pytest.mark.skip(reason="ONNX export fails for tiny-random-phi3 model in current environment")
@pytest.mark.parametrize("quantized", [True, False])
def test_tie_word_embeddings(tmp_path, quantized):
    # setup
    tiny_model = get_tiny_phi3()
    local_tiny_path = tmp_path / "input_model"
    tiny_model_loaded = tiny_model.load_model()
    tiny_model_loaded.tie_weights()
    tiny_model_loaded.config.tie_word_embeddings = True
    tiny_model_loaded.save_pretrained(local_tiny_path)
    tiny_model.save_metadata(local_tiny_path)
    input_model = HfModelHandler(local_tiny_path)

    if quantized:
        input_model = create_pass_from_dict(
            Rtn,
            {"bits": 4, "group_size": 16, "sym": False, "lm_head": True, "embeds": True},
            disable_search=True,
        ).run(input_model, str(tmp_path / "quantized_model"))
    # Use TorchScript exporter because dynamo fails to export this model with TorchExportError
    input_model = create_pass_from_dict(
        OnnxConversion,
        {"torch_dtype": "float32", "use_dynamo_exporter": False},
        disable_search=True,
    ).run(input_model, str(tmp_path / "onnx"))

    output_folder = str(tmp_path / "output")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "TieWordEmbeddings"}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    original_counts = Counter()
    original_dag = OnnxDAG.from_model_path(input_model.model_path)
    for node in original_dag.get_node_names():
        original_counts[original_dag.get_node_op_type(node)] += 1
    new_counts = Counter()
    new_dag = OnnxDAG.from_model_path(output_model.model_path)
    for node in new_dag.get_node_names():
        new_counts[new_dag.get_node_op_type(node)] += 1
    if not quantized:
        assert original_counts["MatMul"] == new_counts["MatMul"] + 1
        assert original_counts["Reshape"] == new_counts["Reshape"] - 2
        assert original_counts["Gemm"] == new_counts["Gemm"] - 1
        assert new_dag.get_node_op_type(new_dag.get_producer("logits")) == "Reshape"
    else:
        assert original_counts["Reshape"] == new_counts["Reshape"] - 1
        assert (
            new_dag.get_node_op_type(new_dag.get_producer(new_dag.get_node_inputs(new_dag.get_producer("logits"))[1]))
            == "Reshape"
        )


def test_remove_gidx_from_matmulnbits(tmp_path):
    # setup
    a_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    b_tensor = numpy_helper.from_array(np.random.randint(0, 255, (3, 3), dtype=np.uint8), name="MatMulNBits.qweight")
    g_idx_sorted = numpy_helper.from_array(np.array([0, 0, 1], dtype=np.int32), name="layers.1.MatMulNBits.g_idx")
    g_idx_random = numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int32), name="layers.2.MatMulNBits.g_idx")
    scale = numpy_helper.from_array(np.array([0.1, 0.15, 0.26], dtype=np.float32), name="MatMulNBits.scales")
    zero_point = numpy_helper.from_array(np.array([128, 128, 128], dtype=np.uint8), name="MatMulNBits.qzeros")

    nodes = [
        # case 1: no gidx provided
        helper.make_node(
            "MatMulNBits",
            name="/layers.0/MatMulNBits",
            inputs=["input", "MatMulNBits.qweight", "MatMulNBits.scales", "MatMulNBits.qzeros"],
            outputs=["output_1"],
            domain="com.microsoft",
            bits=4,
            accuracy_level=4,
            block_size=32,
            K=3,
            N=3,
        ),
        # case 2: sorted gidx provided
        helper.make_node(
            "MatMulNBits",
            name="/layers.1/MatMulNBits",
            inputs=[
                "output_1",
                "MatMulNBits.qweight",
                "MatMulNBits.scales",
                "MatMulNBits.qzeros",
                "layers.1.MatMulNBits.g_idx",
            ],
            outputs=["output_2"],
            domain="com.microsoft",
            bits=4,
            accuracy_level=4,
            block_size=32,
            K=3,
            N=3,
        ),
        # case 3: random gidx provided
        helper.make_node(
            "MatMulNBits",
            name="/layers.2/MatMulNBits",
            inputs=[
                "output_2",
                "MatMulNBits.qweight",
                "MatMulNBits.scales",
                "MatMulNBits.qzeros",
                "layers.2.MatMulNBits.g_idx",
            ],
            outputs=["output"],
            domain="com.microsoft",
            bits=4,
            accuracy_level=4,
            block_size=32,
            K=3,
            N=3,
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=[a_tensor],
        outputs=[output_tensor],
        initializer=[b_tensor, scale, zero_point, g_idx_sorted, g_idx_random],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    input_model = ONNXModelHandler(model_path=str(model_path))
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RemoveGidxFromMatMulNBits"}]},
        disable_search=True,
    )

    output_model = p.run(input_model, output_folder)
    output_model_def = output_model.load_model()
    case_1_ips, case_2_ips, case_3_ips = [node.input for node in output_model_def.graph.node]

    # case 1: No gidx was provided
    assert len(case_1_ips) == 4

    # case 2: Sorted gidx was provided, so it must be removed from node
    assert len(case_2_ips) == 4
    assert "layers.1.MatMulNBits.g_idx" not in case_2_ips

    # case 3: Random gidx was provided, so it should not be removed
    assert len(case_3_ips) == 5
    assert "layers.2.MatMulNBits.g_idx" in case_3_ips


def test_rename_output_dims(tmp_path):
    # setup: create a model with a dynamic dimension in output shape
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["old_dim_name", 3, 4])

    node = helper.make_node("Identity", inputs=["input"], outputs=["output"], name="Identity")

    graph = helper.make_graph(
        nodes=[node],
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    input_model = ONNXModelHandler(model_path=str(model_path))
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameOutputDims", "output_idx": 0, "dim_idx": 0, "dim_name": "new_dim_name"}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)
    output_model_def = output_model.load_model()

    # assert
    output_shape = output_model_def.graph.output[0].type.tensor_type.shape
    dim_names = [dim.dim_param for dim in output_shape.dim]
    assert dim_names[0] == "new_dim_name"


def test_rename_output_dims_invalid_output_idx(tmp_path):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

    graph = helper.make_graph(
        nodes=[node],
        name="TestGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    input_model = ONNXModelHandler(model_path=str(model_path))
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameOutputDims", "output_idx": 5, "dim_idx": 0, "dim_name": "new_name"}]},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="output_idx 5 is out of range"):
        p.run(input_model, output_folder)


def test_packed_attention_to_loop_mha(tmp_path):
    # setup: create model with custom::PackedAttention node
    batch_size, num_heads, seq_len, head_dim = 1, 2, 6, 4

    query = helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim])
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim])
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim])
    cu_seqlens = helper.make_tensor_value_info("cu_seqlens", TensorProto.INT32, [3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, num_heads, head_dim])

    packed_attn_node = helper.make_node(
        OpType.PackedAttention,
        inputs=["query", "key", "value", "cu_seqlens"],
        outputs=["output"],
        name="packed_attention",
        domain=OpType.Custom,
        scale=0.5,
        num_heads=num_heads,
    )

    graph = helper.make_graph(
        nodes=[packed_attn_node],
        name="TestGraph",
        inputs=[query, key, value, cu_seqlens],
        outputs=[output],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid(OpType.Custom, 1)])
    model.ir_version = 10
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    input_model = ONNXModelHandler(model_path=str(model_path))
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "PackedAttentionToLoopMHA"}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)
    output_model_def = output_model.load_model()

    # assert: PackedAttention should be replaced with Loop and MultiHeadAttention
    op_types = [node.op_type for node in output_model_def.graph.node]
    assert OpType.PackedAttention not in op_types
    assert OpType.Loop in op_types

    # MultiHeadAttention is in the Loop's body subgraph
    loop_node = next(node for node in output_model_def.graph.node if node.op_type == OpType.Loop)
    body_graph = loop_node.attribute[0].g
    body_op_types = [node.op_type for node in body_graph.node]
    assert OpType.MultiHeadAttention in body_op_types


def test_packed_attention_to_packed_mha(tmp_path):
    # setup: create model with custom::PackedAttention node
    batch_size, num_heads, seq_len, head_dim = 1, 2, 6, 4

    query = helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim])
    key = helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim])
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim])
    cu_seqlens = helper.make_tensor_value_info("cu_seqlens", TensorProto.INT32, [3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, num_heads, head_dim])

    packed_attn_node = helper.make_node(
        OpType.PackedAttention,
        inputs=["query", "key", "value", "cu_seqlens"],
        outputs=["output"],
        name="packed_attention",
        domain=OpType.Custom,
        scale=0.5,
        num_heads=num_heads,
    )

    graph = helper.make_graph(
        nodes=[packed_attn_node],
        name="TestGraph",
        inputs=[query, key, value, cu_seqlens],
        outputs=[output],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid(OpType.Custom, 1)])
    model.ir_version = 10
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)

    input_model = ONNXModelHandler(model_path=str(model_path))
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "PackedAttentionToPackedMHA"}]},
        disable_search=True,
    )

    # execute
    output_model = p.run(input_model, output_folder)
    output_model_def = output_model.load_model()

    # assert: PackedAttention should be replaced with PackedMultiHeadAttention
    op_types = [node.op_type for node in output_model_def.graph.node]
    assert OpType.PackedAttention not in op_types
    assert OpType.PackedMultiHeadAttention in op_types
