# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.io_update import OpenVINOIoUpdate
from test.utils import get_pytorch_model, get_pytorch_model_dummy_input


def convert_pt_to_ov_model(tmp_path, output_folder=None):
    input_model = get_pytorch_model()
    openvino_conversion_config = {"example_input_func": get_pytorch_model_dummy_input}

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder = output_folder or str(tmp_path / "openvino")

    # execute
    openvino_model = p.run(input_model, output_folder)
    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()

    return openvino_model


def test_openvino_io_update_pass_static(tmp_path):
    # setup
    openvino_model = convert_pt_to_ov_model(tmp_path)
    openvino_conversion_config = {"input_shapes": [[1]], "static": True}

    p = create_pass_from_dict(OpenVINOIoUpdate, openvino_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "openvino_st")

    # execute
    openvino_model = p.run(openvino_model, output_folder)

    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model_st.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model_st.xml").is_file()

    # cleanup
    shutil.rmtree(openvino_model.model_path)


def test_openvino_io_update_pass_dynamic(tmp_path):
    # setup
    openvino_model = convert_pt_to_ov_model(tmp_path)
    openvino_conversion_config = {"input_shapes": [[1]], "static": False}

    p = create_pass_from_dict(OpenVINOIoUpdate, openvino_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "openvino_dy")

    # execute
    openvino_model = p.run(openvino_model, output_folder)

    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model_dy.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model_dy.xml").is_file()

    # cleanup
    shutil.rmtree(openvino_model.model_path)
