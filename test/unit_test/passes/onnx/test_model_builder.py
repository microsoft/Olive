# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest

from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_builder import ModelBuilder


def make_local_model(save_path, model_type="hf"):
    input_model = HfModelHandler(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM")
    loaded_model = input_model.load_model()
    # this checkpoint has an invalid generation config that cannot be saved
    loaded_model.generation_config.pad_token_id = 1

    save_path.mkdir(parents=True, exist_ok=True)
    if model_type == "hf":
        loaded_model.save_pretrained(save_path)
    else:
        onnx_file_path = save_path / "model.onnx"
        onnx_file_path.write_text("dummy onnx file")
        loaded_model.config.save_pretrained(save_path)
        loaded_model.generation_config.save_pretrained(save_path)
    input_model.get_hf_tokenizer().save_pretrained(save_path)

    return (
        HfModelHandler(model_path=save_path)
        if model_type == "hf"
        else ONNXModelHandler(model_path=save_path, onnx_file_name="model.onnx")
    )


@pytest.mark.parametrize("metadata_only", [True, False])
def test_model_builder(tmp_path, metadata_only):
    input_model = make_local_model(tmp_path / "input_model", "onnx" if metadata_only else "hf")

    p = create_pass_from_dict(ModelBuilder, {"precision": "fp32", "metadata_only": metadata_only}, disable_search=True)
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    assert Path(output_folder / "genai_config.json").exists()
