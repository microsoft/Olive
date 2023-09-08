# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_onnx_model
from unittest.mock import patch

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OrtPerfTuning


@pytest.mark.parametrize("config", [{"input_names": ["input"], "input_shapes": [[1, 1]]}, {}])
def test_ort_perf_tuning_pass(config, tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


@patch("olive.model.ONNXModel.get_io_config")
def test_ort_perf_tuning_pass_with_dynamic_shapes(mock_get_io_config, tmp_path):
    mock_get_io_config.return_value = {
        "input_names": ["input"],
        "input_shapes": [["input_0", "input_1"]],
        "input_types": ["float32", "float32"],
        "output_names": ["output"],
        "output_shapes": [["input_0", 10]],
        "output_types": ["float32", "float32"],
    }

    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    with pytest.raises(TypeError) as e:
        # execute
        p.run(input_model, None, output_folder)
        assert "ones() received an invalid combination of arguments" in str(e.value)
