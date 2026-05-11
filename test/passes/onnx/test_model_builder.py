# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import onnx
import pytest

from olive.model import ONNXModelHandler
from olive.model.handler.hf import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_builder import ModelBuilder
from olive.passes.pytorch.rtn import Rtn
from test.utils import make_local_tiny_llama


@pytest.mark.parametrize("metadata_only", [True, False])
def test_model_builder(tmp_path, metadata_only):
    input_model = make_local_tiny_llama(tmp_path / "input_model", "onnx" if metadata_only else "hf")

    p = create_pass_from_dict(
        ModelBuilder,
        {"precision": "fp32", "metadata_only": metadata_only, "extra_options": {"int4_is_symmetric": True}},
        disable_search=True,
    )
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    assert Path(output_folder / "genai_config.json").exists()


@pytest.mark.parametrize("embeds", [True, False])
@pytest.mark.parametrize("group_size", [16, -1])
def test_model_builder_olive_quant(tmp_path, embeds, group_size):
    # set up quantized model
    input_model = create_pass_from_dict(
        Rtn,
        {
            "bits": 4,
            "group_size": group_size,
            "symmetric": False,
            "lm_head": True,
            "embeds": embeds,
        },
        disable_search=True,
    ).run(
        make_local_tiny_llama(tmp_path / "hf_model", "hf"),
        tmp_path / "quantized_model",
    )

    p = create_pass_from_dict(ModelBuilder, {"precision": "int4"}, disable_search=True)
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    assert Path(output_folder / "genai_config.json").exists()


@pytest.mark.parametrize("layer_annotations", [True, False])
def test_model_builder_layer_annotations(tmp_path, layer_annotations):
    """Test that layer annotations are correctly applied to the output ONNX model."""
    input_model = make_local_tiny_llama(tmp_path / "input_model", "hf")

    if layer_annotations:
        # Create layer annotations to be applied
        # Keys are layer names, values are lists of node-name substrings to match
        annotations = {
            "embedding_layer": ["embed_tokens"],
            "norm_layer": ["norm"],
        }
        input_model.model_attributes = {"layer_annotations": annotations}

    p = create_pass_from_dict(
        ModelBuilder,
        {"precision": "fp32"},
        disable_search=True,
    )
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()

    if layer_annotations:
        # Verify that metadata properties were applied to nodes
        model_proto = onnx.load(output_model.model_path, load_external_data=False)
        node_names_with_metadata = {node.name for node in model_proto.graph.node if node.metadata_props}
        assert len(node_names_with_metadata) > 0, (
            "Expected nodes with metadata_props when layer_annotations are provided"
        )


def test_model_builder_uses_saved_test_model_path(tmp_path):
    test_model_path = tmp_path / "saved_test_model"
    output_folder = tmp_path / "output_model"

    mock_cfg = MagicMock()
    mock_cfg.to_dict.return_value = {}
    with patch.object(HfModelHandler, "get_hf_model_config", return_value=mock_cfg):
        input_model = HfModelHandler(
            model_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            test_model_config={"hidden_layers": 2},
            test_model_path=str(test_model_path),
        )

    def materialize_test_model(*_, **__):
        test_model_path.mkdir(parents=True, exist_ok=True)
        (test_model_path / "config.json").write_text("{}")
        return MagicMock()

    def fake_create_model(*_, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / kwargs["filename"]).write_text("dummy onnx file")
        (output_dir / "genai_config.json").write_text("{}")

    fake_builder = types.ModuleType("onnxruntime_genai.models.builder")
    fake_builder.create_model = MagicMock(side_effect=fake_create_model)
    fake_models = types.ModuleType("onnxruntime_genai.models")
    fake_models.builder = fake_builder
    fake_ort_genai = types.ModuleType("onnxruntime_genai")
    fake_ort_genai.models = fake_models
    fake_ort_genai.__version__ = "0.0.0"

    p = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True)

    with (
        patch.object(ModelBuilder, "maybe_patch_quant"),
        patch.dict(
            sys.modules,
            {
                "onnxruntime_genai": fake_ort_genai,
                "onnxruntime_genai.models": fake_models,
                "onnxruntime_genai.models.builder": fake_builder,
            },
        ),
        patch.object(input_model, "load_model", side_effect=materialize_test_model) as mock_load_model,
        patch.object(input_model, "save_metadata", return_value=[]),
    ):
        output_model = p.run(input_model, output_folder)

    assert isinstance(output_model, ONNXModelHandler)
    assert mock_load_model.call_count == 1
    assert Path(output_model.model_path).exists()
    assert test_model_path.exists()
    assert fake_builder.create_model.call_args.kwargs["model_name"] == str(test_model_path)
    assert fake_builder.create_model.call_args.kwargs["input_path"] == str(test_model_path)
