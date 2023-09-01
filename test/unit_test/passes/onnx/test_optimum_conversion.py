import tempfile
from pathlib import Path
from test.unit_test.utils import get_optimum_model_by_hf_config, get_optimum_model_by_model_path

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.optimum_conversion import OptimumConversion
from olive.systems.local import LocalSystem


@pytest.mark.parametrize("input_model", [get_optimum_model_by_hf_config(), get_optimum_model_by_model_path()])
def test_onnx_conversion_pass(input_model):
    # setup
    local_system = LocalSystem()

    p = create_pass_from_dict(OptimumConversion, {}, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = Path(tempdir)

        # The conversion need torch version > 1.13.1, otherwise, it will complain
        # Unsupported ONNX opset version: 18
        onnx_model = local_system.run_pass(p, input_model, None, output_folder)

        # assert
        assert Path(onnx_model.model_path).exists()
