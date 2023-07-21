import tempfile
from pathlib import Path
from test.unit_test.utils import get_pytorch_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.systems.local import LocalSystem


def test_onnx_conversion_pass():
    # setup
    local_system = LocalSystem()
    input_model = get_pytorch_model()

    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        onnx_model = local_system.run_pass(p, input_model, output_folder)

        # assert
        assert Path(onnx_model.model_path).exists()
