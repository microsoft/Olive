import platform
from unittest.mock import patch

import pytest

from olive.common.constants import OS
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qnn.conversion import QNNConversion
from test.utils import get_onnx_model


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
        "1,1",
        "--out_node",
        "output",
    ]
    if config:
        expected_cmd_list.extend(["--show_unconsumed_nodes", "--custom_io", "CUSTOM_IO"])
    mocked_qnn_sdk_runner.return_value.run.assert_called_with(expected_cmd_list)
