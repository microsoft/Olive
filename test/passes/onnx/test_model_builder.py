# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest

from olive.model import ONNXModelHandler
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
