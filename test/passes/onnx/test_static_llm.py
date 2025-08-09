# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.static_llm import StaticLLM
from test.utils import make_local_tiny_llama


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
