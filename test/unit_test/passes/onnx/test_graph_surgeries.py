# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from test.unit_test.utils import make_local_tiny_llama

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries
from olive.passes.onnx.model_builder import ModelBuilder
from olive.passes.onnx.onnx_dag import OnnxDAG


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

    # Construct expected node and initializer names for scale and zero_point
    output_name = "output1"
    scale_node_name = f"{output_name}_exposed_scale"
    scale_initializer_name = f"{scale_node_name}_value"

    zero_point_node_name = f"{output_name}_exposed_zero_point"
    zero_point_initializer_name = f"{zero_point_node_name}_value"

    # Validate that the scale node and its initializer exist in the modified model
    assert any(node.name == scale_node_name for node in output_model.graph.node), "Scale node not added."
    scale_initializer = next(init for init in output_model.graph.initializer if init.name == scale_initializer_name)
    assert np.allclose(
        numpy_helper.to_array(scale_initializer), np.array([original_scale_value], dtype=np.float32)
    ), "Scale value mismatch."

    # Validate that the zero_point node and its initializer exist in the modified model
    assert any(node.name == zero_point_node_name for node in output_model.graph.node), "Zero point node not added."
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
    torch.onnx.export(
        module, torch.randn(1, hidden_size), input_model_path, input_names=["x"], output_names=["y"], opset_version=20
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
            domain="com.microsoft",
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
    tiny_model = HfModelHandler("katuni4ka/tiny-random-phi3")
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
    ]
    output_tensors = [
        helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("output3", TensorProto.FLOAT, [1]),
    ]
    initializers = [
        helper.make_tensor("init", TensorProto.FLOAT, [], [min_value]),
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
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="TestGraph",
        inputs=input_tensors,
        outputs=output_tensors,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
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
        },
    )
    assert all(o == -1e4 for o in outputs)
