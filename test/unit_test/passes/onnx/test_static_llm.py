# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import make_local_tiny_llama

from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.static_llm import StaticLLM


def test_static_llm(tmp_path):
    # setup
    from olive.passes.onnx.model_builder import ModelBuilder
    from olive.passes.onnx.split import SplitModel

    pytorch_model = make_local_tiny_llama(tmp_path)
    onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
        pytorch_model, tmp_path / "onnx_model"
    )

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
    ).run(onnx_model, tmp_path / "split_model")

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
