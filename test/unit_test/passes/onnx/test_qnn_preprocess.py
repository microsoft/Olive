from test.unit_test.utils import get_onnx_model

import pytest
from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import QNNPreprocess


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.17.0"),
    reason="qnn quantization is only supported in onnxruntime>=1.17.0",
)
def test_qnn_preprocess(tmp_path):
    input_model = get_onnx_model()
    p = create_pass_from_dict(QNNPreprocess, {}, disable_search=True)
    out = p.run(input_model, None, tmp_path)
    assert out is not None
