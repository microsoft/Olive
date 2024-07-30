from test.unit_test.utils import get_onnx_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mixed_precision_overrides import MixedPrecisionOverrides


def test_qnn_mixed_precision_overrides(tmp_path):
    input_model = get_onnx_model()
    p = create_pass_from_dict(
        MixedPrecisionOverrides,
        {
            "overrides_config": {
                "/fc1/Gemm_output_0": "QUInt16",
            }
        },
        disable_search=True,
    )
    out = p.run(input_model, tmp_path)
    assert out == input_model
