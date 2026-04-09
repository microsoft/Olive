# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModelHandler
from olive.model.handler.model_package import ModelPackageModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.encapsulation import OpenVINOEncapsulation
from olive.passes.openvino.io_update import OpenVINOIoUpdate
from test.utils import get_pytorch_model, get_pytorch_model_dummy_input

pytestmark = pytest.mark.openvino


def convert_pt_to_ov_model(tmp_path, static=False):
    input_model = get_pytorch_model()
    openvino_conversion_config = {"example_input_func": get_pytorch_model_dummy_input}

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder_convert = str(tmp_path / "openvino_convert")

    # execute
    openvino_model = p.run(input_model, output_folder_convert)
    # assert
    assert Path(openvino_model.model_path).exists()
    assert (Path(openvino_model.model_path) / "ov_model.bin").is_file()
    assert (Path(openvino_model.model_path) / "ov_model.xml").is_file()

    openvino_conversion_config = {"input_shapes": [[1, 1]], "static": static}

    p = create_pass_from_dict(OpenVINOIoUpdate, openvino_conversion_config, disable_search=True)
    output_folder_reshape = str(tmp_path / "openvino_io_update")

    # execute
    openvino_model = p.run(openvino_model, output_folder_reshape)

    assert Path(openvino_model.model_path).exists()
    if static:
        model_name = "ov_model_st"
    else:
        model_name = "ov_model_dy"

    assert (Path(output_folder_reshape) / f"{model_name}.bin").is_file()
    assert (Path(output_folder_reshape) / f"{model_name}.xml").is_file()

    return openvino_model


def test_openvino_encapsulate_pass_static(tmp_path):
    # setup
    openvino_model = convert_pt_to_ov_model(tmp_path, True)
    openvino_conversion_config = {"ov_version": "2025.1"}

    p = create_pass_from_dict(OpenVINOEncapsulation, openvino_conversion_config, disable_search=True)

    # Ensure folder matches reshape pass
    output_folder = str(tmp_path / "openvino_encapsulate")

    # execute
    onnx_model = p.run(openvino_model, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
    assert (Path(onnx_model.model_path)).is_file()


def test_openvino_encapsulate_pass_dynamic(tmp_path):
    # setup
    openvino_model = convert_pt_to_ov_model(tmp_path)
    openvino_conversion_config = {"target_device": "npu"}

    p = create_pass_from_dict(OpenVINOEncapsulation, openvino_conversion_config, disable_search=True)

    # Ensure folder name is unique
    output_folder = str(tmp_path / "openvino_encapsulate")

    # execute
    onnx_model = p.run(openvino_model, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
    assert (Path(onnx_model.model_path)).is_file()


def test_openvino_encapsulate_pass_dynamic_keep_ov_dynamic_dims(tmp_path):
    # setup
    openvino_model = convert_pt_to_ov_model(tmp_path)
    openvino_conversion_config = {"target_device": "npu", "keep_ov_dynamic_dims": True}

    p = create_pass_from_dict(OpenVINOEncapsulation, openvino_conversion_config, disable_search=True)

    # Ensure folder name is unique
    output_folder = str(tmp_path / "openvino_encapsulate")

    # execute
    onnx_model = p.run(openvino_model, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
    assert (Path(onnx_model.model_path)).is_file()


# ===========================================================================
# Model package tests
# ===========================================================================


def test_model_package_returns_model_package_handler(tmp_path):
    accelerator_spec = AcceleratorSpec(accelerator_type=Device.NPU, execution_provider="OpenVINOExecutionProvider")

    p = create_pass_from_dict(
        OpenVINOEncapsulation,
        {"ov_version": ["2025.1", "2025.2"], "target_device": "npu"},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    with patch.object(OpenVINOEncapsulation, "_run_single_target") as mock_single:

        def side_effect(model, config, output_model_path):
            out_dir = Path(output_model_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            model_file = out_dir / "model.onnx"
            model_file.write_text("dummy")
            return ONNXModelHandler(
                model_path=str(model_file),
                model_attributes={
                    "ep": "OpenVINOExecutionProvider",
                    "device": "NPU",
                    "sdk_version": config.ov_version,
                    "architecture": "NPU",
                },
            )

        mock_single.side_effect = side_effect

        input_model = MagicMock()
        input_model.model_attributes = {}
        output_path = str(tmp_path / "output.onnx")
        result = p.run(input_model, output_path)

    assert isinstance(result, ModelPackageModelHandler)
    assert result.target_names == ["ov_2025_1", "ov_2025_2"]
    assert mock_single.call_count == 2


def test_single_target_populates_model_attributes(tmp_path):
    accelerator_spec = AcceleratorSpec(accelerator_type=Device.NPU, execution_provider="OpenVINOExecutionProvider")

    p = create_pass_from_dict(
        OpenVINOEncapsulation,
        {"ov_version": "2025.1", "target_device": "npu"},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    with patch.object(OpenVINOEncapsulation, "_run_single_target") as mock_single:

        def side_effect(model, config, output_model_path):
            out_dir = Path(output_model_path)
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            model_file = out_dir / "model.onnx"
            model_file.write_text("dummy")
            return ONNXModelHandler(
                model_path=str(model_file),
                model_attributes={
                    "ep": "OpenVINOExecutionProvider",
                    "device": "NPU",
                    "sdk_version": "2025.1",
                    "architecture": "NPU",
                },
            )

        mock_single.side_effect = side_effect

        input_model = MagicMock()
        input_model.model_attributes = {}
        output_path = str(tmp_path / "output.onnx")
        result = p.run(input_model, output_path)

    assert isinstance(result, ONNXModelHandler)
    assert result.model_attributes["ep"] == "OpenVINOExecutionProvider"
    assert result.model_attributes["sdk_version"] == "2025.1"
