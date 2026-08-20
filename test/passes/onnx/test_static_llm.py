# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path

import onnx
import pytest

from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.static_llm import StaticLLM
from test.utils import make_local_tiny_llama


def _assert_static_io_shapes(model: ONNXModelHandler):
    model_proto = onnx.load(model.model_path, load_external_data=False)

    for value_info in [*model_proto.graph.input, *model_proto.graph.output]:
        shape = value_info.type.tensor_type.shape
        dynamic_dims = [dim.dim_param or "<unknown>" for dim in shape.dim if not dim.HasField("dim_value")]
        assert not dynamic_dims, f"{value_info.name} has dynamic dimensions: {dynamic_dims}"


def test_static_llm(tmp_path):
    # setup
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

    split_model = create_pass_from_dict(
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

        fixed_context_len_model = create_pass_from_dict(
            DynamicToFixedShape,
            {"dim_param": ["past_sequence_length"], "dim_value": [4096]},
            disable_search=True,
        ).run(post_op_model, setup_path / "fixed_context_len_model")

        return fixed_context_len_model

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
                "update_genai_config": True,
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
        assert set(genai_config["model"]["decoder"]["pipeline"][0].keys()) == {"prefill", "decode"}
        assert not genai_config["model"]["decoder"]["pipeline"][0]["prefill"]["run_on_token_gen"]
        assert not genai_config["model"]["decoder"]["pipeline"][0]["decode"]["run_on_prompt"]
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
                "update_genai_config": True,
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
        assert set(pipeline_config) == {"model"}
        assert pipeline_config["model"]["filename"] == "model.onnx"
        assert genai_config["model"]["context_length"] == 4096
        assert genai_config["search"]["max_length"] == 4096

