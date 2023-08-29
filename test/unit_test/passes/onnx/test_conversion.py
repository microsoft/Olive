import tempfile
from pathlib import Path
from test.unit_test.utils import get_hf_model_with_past, get_pytorch_model

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.systems.local import LocalSystem


@pytest.mark.parametrize("input_model", [get_pytorch_model(), get_hf_model_with_past()])
def test_onnx_conversion_pass(input_model):
    # setup
    local_system = LocalSystem()

    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # The conversion need torch version > 1.13.1, otherwise, it will complain
        # Unsupported ONNX opset version: 18
        onnx_model = local_system.run_pass(p, input_model, None, output_folder)

        # assert
        assert Path(onnx_model.model_path).exists()
