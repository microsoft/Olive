import platform
from test.unit_test.utils import get_onnx_model
from unittest.mock import patch

import pytest

from olive.common.constants import OS
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qnn.conversion import QNNConversion


@patch("olive.passes.qnn.conversion.QNNSDKRunner")
@pytest.mark.parametrize(
    "config",
    [
        {
            "input_dim": ["input 1,1"],
            "out_node": ["output"],
            "extra_args": "--show_unconsumed_nodes --custom_io CUSTOM_IO",
        },
        {},
    ],
)
def test_qnn_conversion_cmd(mocked_qnn_sdk_runner, config, tmp_path):
    input_model = get_onnx_model()
    p = create_pass_from_dict(QNNConversion, config, disable_search=True)
    mocked_qnn_sdk_runner.return_value.sdk_env.sdk_root_path = "sdk_root_path"
    mocked_qnn_sdk_runner.return_value.sdk_env.target_arch = "x86_64"
    p.run(input_model, tmp_path)
    converter_program = ["qnn-onnx-converter"]
    if platform.system() == OS.WINDOWS:
        converter_program = ["python", "sdk_root_path\\bin\\x86_64\\qnn-onnx-converter"]

    expected_cmd_list = [
        *converter_program,
        "--input_network",
        input_model.model_path,
        "--output_path",
        str(tmp_path / "model.cpp"),
        "--input_dim",
        "input",
        "1,1",  # got from onnx model graph even if no input_dim in config
        "--out_node",
        "output",  # got from onnx model graph even if no out_node in config
    ]
    if config:
        expected_cmd_list.extend(["--show_unconsumed_nodes", "--custom_io", "CUSTOM_IO"])
    mocked_qnn_sdk_runner.return_value.run.assert_called_with(expected_cmd_list)
