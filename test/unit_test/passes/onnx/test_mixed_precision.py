from pathlib import Path
import tempfile
from olive.model import ONNXModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.mixed_precision import OrtMixedPrecision
from olive.systems.local import LocalSystem
from test.unit_test.utils import get_onnx_model


def test_ort_mixed_precision_pass():
    # setup
    local_system = LocalSystem()
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtMixedPrecision, {}, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, input_model, output_folder)
