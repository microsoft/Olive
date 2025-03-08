# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import make_local_tiny_llama

import pytest

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_builder import ModelBuilder


@pytest.mark.parametrize("metadata_only", [True, False])
def test_model_builder(tmp_path, metadata_only):
    input_model = make_local_tiny_llama(tmp_path / "input_model", "onnx" if metadata_only else "hf")

    p = create_pass_from_dict(ModelBuilder, {"precision": "fp32", "metadata_only": metadata_only}, disable_search=True)
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    assert Path(output_folder / "genai_config.json").exists()
