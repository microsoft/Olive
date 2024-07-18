import shutil
from test.unit_test.utils import get_onnx_model
from unittest.mock import patch

import pytest
from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.qnn.qnn_preprocess import QNNPreprocess


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.17.0"),
    reason="qnn quantization is only supported in onnxruntime>=1.17.0",
)
def test_qnn_preprocess_not_change_model(tmp_path):
    input_model = get_onnx_model()
    p = create_pass_from_dict(QNNPreprocess, {}, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out == input_model


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.17.0"),
    reason="skip model is modified case as qnn quantization is only supported in onnxruntime>=1.17.0",
)
@patch("onnxruntime.quantization.execution_providers.qnn.qnn_preprocess_model")
def test_qnn_preprocess_changed_model(mocked_qnn_preprocess_model, tmp_path):
    mocked_qnn_preprocess_model.return_value = True
    input_model = get_onnx_model()
    shutil.copy(input_model.model_path, tmp_path / "dummy_model.onnx")
    p = create_pass_from_dict(QNNPreprocess, {}, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out != input_model


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.18.0"),
    reason="qnn quantization extra configs is only supported in onnxruntime>=1.18.0",
)
def test_qnn_preprocess_extra_configs(tmp_path):
    input_model = get_onnx_model()
    p = create_pass_from_dict(
        QNNPreprocess,
        {
            "save_as_external_data": False,
            "convert_attribute": None,
            "inputs_to_make_channel_last": None,
            "outputs_to_make_channel_last": None,
        },
        disable_search=True,
    )
    out = p.run(input_model, tmp_path)
    assert out == input_model
