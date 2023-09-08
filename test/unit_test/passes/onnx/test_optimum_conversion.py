from pathlib import Path
from test.unit_test.utils import get_optimum_model_by_hf_config, get_optimum_model_by_model_path

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.optimum_conversion import OptimumConversion


@pytest.mark.parametrize("input_model", [get_optimum_model_by_hf_config(), get_optimum_model_by_model_path()])
def test_optimum_conversion_pass(input_model, tmp_path):
    # setup
    p = create_pass_from_dict(OptimumConversion, {}, disable_search=True)
    output_folder = tmp_path

    # execute
    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
