# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import make_local_tiny_llama

import pytest

from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.compose import ComposeOnnxModels
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.model_builder import ModelBuilder
from olive.passes.onnx.split import SplitModel
from olive.passes.onnx.static_llm import StaticLLM


@pytest.mark.parametrize("use_mb", [True, False])
def test_compose_onnx_models_composite(tmp_path, use_mb):
    # setup
    pytorch_model = make_local_tiny_llama(tmp_path)
    onnx_model = create_pass_from_dict(
        ModelBuilder if use_mb else OnnxConversion, {"precision": "fp32"} if use_mb else {}, disable_search=True
    ).run(pytorch_model, tmp_path / "onnx_model")
    split_model = create_pass_from_dict(
        SplitModel,
        {
            "split_assignments": {
                "model.embed_tokens": 0,
                "model.layers.0": 1,
                "model.layers.1": 2,
                "lm_head": 3,
            }
        },
        disable_search=True,
    ).run(onnx_model, tmp_path / "split_model")

    p = create_pass_from_dict(ComposeOnnxModels, {}, disable_search=True)

    # execute
    output_model_path = tmp_path / "output_model"
    output_model = p.run(split_model, output_model_path)

    # check
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    expected_io_config = onnx_model.io_config
    actual_io_config = output_model.io_config
    for io_key in ["input", "output"]:
        # check the i/o names match
        assert set(actual_io_config[f"{io_key}_names"]) == set(expected_io_config[f"{io_key}_names"])
        # check the i/o shapes/types match
        for info_key in ["shapes", "types"]:
            expected_info_dict = dict(
                zip(expected_io_config[f"{io_key}_names"], expected_io_config[f"{io_key}_{info_key}"])
            )
            actual_info_dict = dict(zip(actual_io_config[f"{io_key}_names"], actual_io_config[f"{io_key}_{info_key}"]))
            for name, value in expected_info_dict.items():
                assert actual_info_dict[name] == value


def test_compose_onnx_models_llm_pipeline(tmp_path):
    # setup
    pytorch_model = make_local_tiny_llama(tmp_path)
    onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
        pytorch_model, tmp_path / "onnx_model"
    )
    split_model = create_pass_from_dict(
        SplitModel,
        {
            "split_assignments": {
                "model.embed_tokens": 0,
                "model.layers.0": 1,
                "model.layers.1": 2,
                "lm_head": 3,
            }
        },
        disable_search=True,
    ).run(onnx_model, tmp_path / "split_model")
    llm_model = create_pass_from_dict(StaticLLM, {"batch_size": 1, "context_length": 64}, disable_search=True).run(
        split_model, tmp_path / "llm_model"
    )

    p = create_pass_from_dict(ComposeOnnxModels, {}, disable_search=True)

    # execute
    output_model_path = tmp_path / "output_model"
    output_model = p.run(llm_model, output_model_path)

    # check
    assert isinstance(output_model, CompositeModelHandler)
    assert output_model.model_attributes["llm_pipeline"] == {
        "embeddings": "embeddings",
        "context": ["context"],
        "iterator": ["iterator"],
        "lm_head": "lm_head",
    }
    assert len(list(output_model.model_components)) == 4
