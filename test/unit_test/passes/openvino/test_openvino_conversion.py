# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from test.unit_test.utils import get_pytorch_model, get_pytorch_model_dummy_input

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion


def test_openvino_conversion_pass(tmp_path):
    # setup
    input_model = get_pytorch_model()
    openvino_conversion_config = {"example_input_func": get_pytorch_model_dummy_input}

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "openvino")

    # execute
    openvino_model = p.run(input_model, output_folder)

    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()

    # cleanup
    shutil.rmtree(openvino_model.model_path)


def test_openvino_conversion_pass_no_example_input(tmp_path):
    # setup
    input_model = get_pytorch_model()
    openvino_conversion_config = {
        "input": [1, 1],
    }

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "openvino")

    # execute
    openvino_model = p.run(input_model, output_folder)

    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()

    # cleanup
    shutil.rmtree(openvino_model.model_path)


def test_openvino_conversion_pass_example_input_with_us(tmp_path):
    # setup
    input_model = get_pytorch_model()
    user_script_path = str(Path(__file__).parent / "user_script.py")
    openvino_conversion_config = {"user_script": user_script_path, "example_input_func": "get_dummy_input"}

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "openvino")

    # execute
    openvino_model = p.run(input_model, output_folder)

    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()

    # cleanup
    shutil.rmtree(openvino_model.model_path)


def test_openvino_conversion_pass_input_with_us(tmp_path):
    # setup
    input_model = get_pytorch_model()
    user_script_path = str(Path(__file__).parent / "user_script.py")
    openvino_conversion_config = {"user_script": user_script_path, "input": "get_input"}

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "openvino")

    # execute
    openvino_model = p.run(input_model, output_folder)

    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()

    # cleanup
    shutil.rmtree(openvino_model.model_path)
