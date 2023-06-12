# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from test.unit_test.utils import get_pytorch_model, get_pytorch_model_dummy_input

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.systems.local import LocalSystem


def test_openvino_conversion_pass():
    # setup
    local_system = LocalSystem()
    input_model = get_pytorch_model()
    dummy_input = get_pytorch_model_dummy_input()
    openvino_conversion_config = {"extra_config": {"example_input": dummy_input}}

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "openvino")

        # execute
        openvino_model = local_system.run_pass(p, input_model, output_folder)

        # assert
        assert Path(openvino_model.model_path).exists()
        assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
        assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()


def test_openvino_conversion_pass_no_example_input():
    # setup
    local_system = LocalSystem()
    input_model = get_pytorch_model()
    openvino_conversion_config = {
        "input_shape": [1, 1],
    }

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "openvino")

        # execute
        openvino_model = local_system.run_pass(p, input_model, output_folder)

        # assert
        assert Path(openvino_model.model_path).exists()
        assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
        assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()
