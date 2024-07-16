import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.common.constants import OS
from olive.model import SNPEModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qnn.context_binary_generator import QNNContextBinaryGenerator


@pytest.mark.skipif(platform.system() == OS.WINDOWS, reason="QNNContextBinaryGenerator is not supported on Windows.")
@patch("olive.passes.qnn.context_binary_generator.QNNSDKRunner")
@pytest.mark.parametrize(
    "config",
    [
        {
            "backend": "libQnnHtp.so",
        },
        {
            "backend": "libQnnHtp.so",
            "extra_args": "--dlc_path extra_test.dlc",
        },
    ],
)
def test_snpe_model_input(mocked_qnn_sdk_runner, config, tmp_path):
    tmp_model_path = tmp_path / "test.dlc"
    tmp_model_path.touch()

    input_model = SNPEModelHandler(
        input_names=["input"],
        input_shapes=[[1, 1]],
        output_names=["output"],
        output_shapes=[[1, 1]],
        model_path=tmp_model_path,
    )
    p = create_pass_from_dict(QNNContextBinaryGenerator, config, disable_search=True)
    mocked_qnn_sdk_runner.return_value.sdk_env.sdk_root_path = "sdk_root_path"
    mocked_qnn_sdk_runner.return_value.sdk_env.target_arch = "x86_64"
    with patch.object(Path, "is_file") as mock_is_file, patch.object(Path, "exists") as mock_exists:
        mock_is_file.return_value = True
        mock_exists.return_value = True
        p.run(input_model, tmp_path)

    extra_args = config.get("extra_args") or ""
    if "dlc_path" not in extra_args:
        assert f"--dlc_path {tmp_model_path}" in mocked_qnn_sdk_runner.return_value.run.call_args[0][0]
    else:
        assert "--dlc_path extra_test.dlc" in mocked_qnn_sdk_runner.return_value.run.call_args[0][0]
