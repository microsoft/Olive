# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from test.unit_test.utils import get_hf_model_with_past, get_onnx_model, get_pytorch_model

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion, OnnxOpVersionConversion


@pytest.mark.parametrize("input_model", [get_pytorch_model(), get_hf_model_with_past()])
def test_onnx_conversion_pass(input_model, tmp_path):
    # setup
    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # The conversion need torch version > 1.13.1, otherwise, it will complain
    # Unsupported ONNX opset version: 18
    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()


@pytest.mark.parametrize("target_opset", [9, 13, 16])
def test_onnx_op_version_conversion_pass(target_opset, tmp_path):
    input_model = get_onnx_model()
    # setup
    p = create_pass_from_dict(OnnxOpVersionConversion, {"target_opset": target_opset}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert onnx_model.load_model().opset_import[0].version == target_opset
