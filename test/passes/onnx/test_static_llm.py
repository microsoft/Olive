# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import onnx
import onnx_ir as ir
import pytest
from onnx import TensorProto, helper

from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.static_llm import StaticLLM, _get_transformer_dim_source
from test.utils import make_local_tiny_llama


def _assert_static_io_shapes(model: ONNXModelHandler):
    model_proto = onnx.load(model.model_path, load_external_data=False)

    for value_info in [*model_proto.graph.input, *model_proto.graph.output]:
        shape = value_info.type.tensor_type.shape
        dynamic_dims = [dim.dim_param or "<unknown>" for dim in shape.dim if not dim.HasField("dim_value")]
        assert not dynamic_dims, f"{value_info.name} has dynamic dimensions: {dynamic_dims}"


def _hidden_state_shape(model_handler):
    """Return the shape of the first rank-3 graph input, i.e. the hidden state entering the component."""
    model_proto = onnx.load(model_handler.model_path, load_external_data=False)
    for value_info in model_proto.graph.input:
        shape = [d.dim_param if d.dim_param else d.dim_value for d in value_info.type.tensor_type.shape.dim]
        if len(shape) == 3:
            return shape
    return None


def test_get_transformer_dim_source_skips_non_hidden_state_input():
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"])
    hidden = helper.make_tensor_value_info("decoder_input", TensorProto.FLOAT, ["batch", "sequence", "hidden_size"])
    graph = helper.make_graph([], "transformer", [mask, hidden], [hidden])
    model = ir.from_proto(helper.make_model(graph))

    assert _get_transformer_dim_source(list(model.graph.inputs)).name == "decoder_input"


def _make_split_model(tmp_path):
    """Build a split tiny-llama composite: [embeddings, layers.0, layers.1, lm_head]."""
    from olive.passes.onnx.graph_surgeries import GraphSurgeries
    from olive.passes.onnx.model_builder import ModelBuilder
    from olive.passes.onnx.split import SplitModel

    pytorch_model = make_local_tiny_llama(tmp_path / "input_model")
    onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
        pytorch_model, tmp_path / "onnx_model"
    )
    post_op_model = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "AttentionMaskToSequenceLengths"}]}, disable_search=True
    ).run(onnx_model, tmp_path / "post_op_model")

    return create_pass_from_dict(
        SplitModel,
        {
            "split_assignments": {
                "model.embed_tokens": 0,
                "model.attn_mask_reformat": 0,
                "model.layers.0": 1,
                "model.layers.1": 2,
                "lm_head": 3,
            }
        },
        disable_search=True,
    ).run(post_op_model, tmp_path / "split_model")


def test_static_llm(tmp_path):
    # setup
    split_model = _make_split_model(tmp_path)

    p = create_pass_from_dict(StaticLLM, {"batch_size": 1, "context_length": 64}, disable_search=True)

    # run
    output_model_path = tmp_path / "output_model"
    output_model = p.run(split_model, output_model_path)

    # check
    assert isinstance(output_model, CompositeModelHandler)
    model_components = list(output_model.model_components)
    assert all(isinstance(m, ONNXModelHandler) for m in model_components)
    assert len(model_components) == 6
    assert output_model.model_attributes["llm_pipeline"] == {
        "embeddings": "embeddings",
        "context": ["context_0", "context_1"],
        "iterator": ["iterator_0", "iterator_1"],
        "lm_head": "lm_head",
    }
    with (output_model_path / "genai_config.json").open() as f:
        genai_config = json.load(f)
    assert genai_config["model"]["type"] == "decoder-pipeline"
    for i_name in ["input_ids", "past_sequence_length"]:
        assert i_name in genai_config["model"]["decoder"]["inputs"]
    assert genai_config["model"]["decoder"]["sliding_window"]["window_size"] == 64
    assert set(genai_config["model"]["decoder"]["pipeline"][0].keys()) == set(output_model.model_component_names)
    assert not genai_config["model"]["decoder"]["pipeline"][0]["context_0"]["run_on_token_gen"]
    assert not genai_config["model"]["decoder"]["pipeline"][0]["iterator_0"]["run_on_prompt"]


def test_static_llm_omits_embeddings_when_no_embeddings_component(tmp_path):
    # setup: drop the embeddings component so the model enters the pipeline at "inputs_embeds"
    split_model = _make_split_model(tmp_path)
    no_embeddings_model = split_model.select_components(list(split_model.model_component_names)[1:])

    p = create_pass_from_dict(StaticLLM, {"batch_size": 1, "context_length": 64}, disable_search=True)

    # run
    output_model_path = tmp_path / "output_model_no_embeddings"
    output_model = p.run(no_embeddings_model, output_model_path)

    # check
    assert isinstance(output_model, CompositeModelHandler)
    model_components = list(output_model.model_components)
    assert all(isinstance(m, ONNXModelHandler) for m in model_components)
    # context_0/1, iterator_0/1, lm_head -- no embeddings component
    assert len(model_components) == 5
    assert output_model.model_attributes["llm_pipeline"] == {
        "context": ["context_0", "context_1"],
        "iterator": ["iterator_0", "iterator_1"],
        "lm_head": "lm_head",
    }
    # the transformer components are still shape-fixed off their own dim params
    components = dict(output_model.get_model_components())
    assert _hidden_state_shape(components["context_0"])[:2] == [1, 64]
    assert _hidden_state_shape(components["iterator_0"])[:2] == [1, 1]


@pytest.mark.parametrize("has_embeddings", [False, True])
@pytest.mark.parametrize("decoder_has_input_ids", [False, True])
def test_static_llm_keeps_transformer_with_input_ids_in_pipeline(tmp_path, has_embeddings, decoder_has_input_ids):
    components = []
    names = []
    if has_embeddings:
        graph = helper.make_graph(
            [helper.make_node("Gather", ["table", "input_ids"], ["inputs_embeds"], axis=0)],
            "embedding",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
            [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["batch", "seq", 8])],
            initializer=[helper.make_tensor("table", TensorProto.FLOAT, [4, 8], [1.0] * 32)],
        )
        path = tmp_path / "embedding.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        components.append(ONNXModelHandler(path))
        names.append("embedding")
    for idx in range(2):
        input_name = "inputs_embeds" if idx == 0 else "hidden_0"
        inputs = [
            helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", "seq", 8]),
            helper.make_tensor_value_info(f"past_key_{idx}", TensorProto.FLOAT, ["batch", 2, 16, 4]),
            helper.make_tensor_value_info(f"past_value_{idx}", TensorProto.FLOAT, ["batch", 2, 16, 4]),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ["batch"]),
            helper.make_tensor_value_info("total_seq_len", TensorProto.INT32, [1]),
        ]
        if decoder_has_input_ids and idx == 0:
            inputs.append(helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]))
        graph = helper.make_graph(
            [
                helper.make_node(
                    "GroupQueryAttention",
                    [
                        input_name,
                        input_name,
                        input_name,
                        f"past_key_{idx}",
                        f"past_value_{idx}",
                        "seqlens_k",
                        "total_seq_len",
                    ],
                    [f"hidden_{idx}", f"present_key_{idx}", f"present_value_{idx}"],
                    domain="com.microsoft",
                    num_heads=2,
                    kv_num_heads=2,
                )
            ],
            f"transformer_{idx}",
            inputs,
            [
                helper.make_tensor_value_info(f"hidden_{idx}", TensorProto.FLOAT, ["batch", "seq", 8]),
                helper.make_tensor_value_info(f"present_key_{idx}", TensorProto.FLOAT, ["batch", 2, 16, 4]),
                helper.make_tensor_value_info(f"present_value_{idx}", TensorProto.FLOAT, ["batch", 2, 16, 4]),
            ],
        )
        path = tmp_path / f"transformer_{idx}.onnx"
        onnx.save(
            helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
            ),
            path,
        )
        components.append(ONNXModelHandler(path))
        names.append(f"transformer_{idx}")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["hidden_1"], ["logits"])],
        "lm_head",
        [helper.make_tensor_value_info("hidden_1", TensorProto.FLOAT, ["batch", "seq", 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "seq", 8])],
    )
    path = tmp_path / "lm_head.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    components.append(ONNXModelHandler(path))
    names.append("lm_head")
    model = CompositeModelHandler(components, names)
    output = create_pass_from_dict(StaticLLM, {"context_length": 64}, disable_search=True).run(
        model, tmp_path / "static"
    )
    pipeline = output.model_attributes["llm_pipeline"]
    assert ("embeddings" in pipeline) == has_embeddings
    assert len(pipeline["context"]) == 2
    components = dict(output.get_model_components())
    assert _hidden_state_shape(components["context_0"])[:2] == [1, 64]
    assert _hidden_state_shape(components["iterator_0"])[:2] == [1, 1]


class TestStaticLlmQnnGpu:
    @pytest.fixture(scope="class")
    def setup_model(self, tmp_path_factory):
        from olive.passes.onnx.dynamic_to_fixed_shape import DynamicToFixedShape
        from olive.passes.onnx.graph_surgeries import GraphSurgeries
        from olive.passes.onnx.model_builder import ModelBuilder

        setup_path = tmp_path_factory.mktemp("static_llm_qnn_gpu")

        pytorch_model = make_local_tiny_llama(setup_path / "input_model")
        onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
            pytorch_model, setup_path / "onnx_model"
        )
        post_op_model = create_pass_from_dict(
            GraphSurgeries,
            {
                "surgeries": [
                    {"surgeon": "RemoveRopeMultiCache"},
                    {"surgeon": "AttentionMaskToSequenceLengths"},
                    {"surgeon": "SimplifiedLayerNormToRMSNorm"},
                ]
            },
            disable_search=True,
        ).run(onnx_model, setup_path / "post_op_model")

        return create_pass_from_dict(
            DynamicToFixedShape,
            {"dim_param": ["past_sequence_length"], "dim_value": [4096]},
            disable_search=True,
        ).run(post_op_model, setup_path / "fixed_context_len_model")

    def test_prefill_decode_models(self, setup_model, tmp_path):
        accelerator_spec = AcceleratorSpec(
            accelerator_type=Device.GPU,
            execution_provider=ExecutionProvider.QNNExecutionProvider,
        )

        p = create_pass_from_dict(
            StaticLLM,
            {
                "batch_size": 1,
                "context_length": 128,
                "prefill_decode_models": True,
            },
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        # run
        output_model_path = tmp_path / "output_model"
        output_model = p.run(setup_model, output_model_path)

        # check
        assert isinstance(output_model, CompositeModelHandler)
        model_components = list(output_model.model_components)
        assert all(isinstance(m, ONNXModelHandler) for m in model_components)
        assert len(model_components) == 2
        with (output_model_path / "genai_config.json").open() as f:
            genai_config = json.load(f)
        assert genai_config["model"]["type"] == "decoder-pipeline"
        for i_name in ["input_ids", "past_sequence_length"]:
            assert i_name in genai_config["model"]["decoder"]["inputs"]
        assert genai_config["model"]["decoder"]["sliding_window"]["window_size"] == 128

        pipeline_config = genai_config["model"]["decoder"]["pipeline"][0]
        assert set(pipeline_config.keys()) == {"prefill", "decode"}
        assert not pipeline_config["prefill"]["run_on_token_gen"]
        assert not pipeline_config["decode"]["run_on_prompt"]
        assert pipeline_config["prefill"]["inherit_session_options"]
        assert pipeline_config["decode"]["inherit_session_options"]
        assert genai_config["model"]["context_length"] == 4096
        assert genai_config["search"]["max_length"] == 4096

        for model_component in model_components:
            _assert_static_io_shapes(model_component)

    def test_single_model(self, setup_model, tmp_path):
        accelerator_spec = AcceleratorSpec(
            accelerator_type=Device.GPU,
            execution_provider=ExecutionProvider.QNNExecutionProvider,
        )

        p = create_pass_from_dict(
            StaticLLM,
            {
                "batch_size": 1,
                "context_length": 128,
                "prefill_decode_models": False,
            },
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        # run
        output_model_path = tmp_path / "output_model"
        output_model = p.run(setup_model, output_model_path)

        # check
        assert isinstance(output_model, ONNXModelHandler)
        _assert_static_io_shapes(output_model)

        with (output_model_path / "genai_config.json").open() as f:
            genai_config = json.load(f)
        assert genai_config["model"]["type"] == "decoder-pipeline"
        for i_name in ["input_ids", "past_sequence_length"]:
            assert i_name in genai_config["model"]["decoder"]["inputs"]
        assert genai_config["model"]["decoder"]["sliding_window"]["window_size"] == 128
        pipeline_config = genai_config["model"]["decoder"]["pipeline"][0]
        assert set(pipeline_config.keys()) == {"model"}
        assert pipeline_config["model"]["filename"] == "model.onnx"
        assert pipeline_config["model"]["inherit_session_options"]
        assert genai_config["model"]["context_length"] == 4096
        assert genai_config["search"]["max_length"] == 4096
