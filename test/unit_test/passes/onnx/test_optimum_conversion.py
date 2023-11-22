# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from test.unit_test.utils import get_optimum_model_by_hf_config, get_optimum_model_by_model_path

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.optimum_conversion import OptimumConversion


@pytest.mark.parametrize(
    "input_model,extra_args",
    [(get_optimum_model_by_hf_config(), {"atol": 0.1}), (get_optimum_model_by_model_path(), {"atol": None})],
)
def test_optimum_conversion_pass(input_model, extra_args, tmp_path):
    # setup
    p = create_pass_from_dict(OptimumConversion, {"extra_args": extra_args}, disable_search=True)
    output_folder = tmp_path

    # execute
    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()


@pytest.mark.parametrize(
    "config,is_valid",
    [
        ({"fp16": True}, False),
        ({"fp16": True, "device": "cpu"}, False),
        ({"fp16": False, "device": "cpu"}, True),
    ],
)
def test_optimum_configs(config, is_valid, tmp_path):
    input_model = get_optimum_model_by_hf_config()
    # setup
    p = create_pass_from_dict(OptimumConversion, config, disable_search=True)
    output_folder = tmp_path

    if not is_valid:
        assert p.validate_search_point(config, None) is False
        with pytest.raises(
            ValueError,
            match="FP16 export is supported only when exporting on GPU. Please pass the option `--device cuda`.",
        ):
            p.run(input_model, None, output_folder)
    else:
        assert p.validate_search_point(config, None)
        onnx_model = p.run(input_model, None, output_folder)
        assert Path(onnx_model.model_path).exists()
