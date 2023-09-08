from test.unit_test.utils import get_onnx_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mixed_precision import OrtMixedPrecision


def test_ort_mixed_precision_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtMixedPrecision, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)
