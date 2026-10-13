# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.encapsulation import OpenVINOEncapsulation
from olive.passes.openvino.io_update import OpenVINOIoUpdate
from olive.passes.openvino.ov_utils import apply_genai_overrides
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


def test_update_genai_config_preserves_package_and_source_hardlink(tmp_path):
    source_config = tmp_path / "genai_config.json"
    original = {
        "model": {
            "type": "gemma4",
            "decoder": {
                "filename": "decoder/model.onnx",
                "inputs": {"inputs_embeds": "inputs_embeds"},
                "outputs": {"logits": "logits"},
                "session_options": {"intra_op_num_threads": 2},
            },
            "embedding": {"filename": "embedding/model.onnx"},
        },
        "tokenizer": {"pad_token_id": 0},
    }
    source_config.write_text(json.dumps(original), encoding="utf-8")
    source_bytes = source_config.read_bytes()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_path = output_dir / "model_ctx.onnx"
    output_path.touch()
    output_config = output_dir / "genai_config.json"
    os.link(source_config, output_config)
    overrides = {
        "model": {
            "decoder": {"session_options": {"intra_op_num_threads": 4}},
            "vision": {"filename": "vision_encoder/model.onnx"},
        },
        "search": {"past_present_share_buffer": False},
    }
    original_overrides = deepcopy(overrides)
    encapsulation = create_pass_from_dict(
        OpenVINOEncapsulation,
        {"update_genai_config": True, "target_device": "npu", "genai_config_override": overrides},
        disable_search=True,
    )
    input_model = MagicMock()
    input_model.model_attributes = {"additional_files": [str(source_config)]}
    output_model = ONNXModelHandler(model_path=str(output_path))

    with patch.object(encapsulation, "_run_single_target", return_value=output_model):
        result = encapsulation.run(input_model, str(output_dir))

    updated = json.loads(output_config.read_text(encoding="utf-8"))
    assert updated["model"]["type"] == original["model"]["type"]
    assert updated["model"]["embedding"] == original["model"]["embedding"]
    assert updated["tokenizer"] == original["tokenizer"]
    decoder = updated["model"]["decoder"]
    assert decoder["inputs"] == original["model"]["decoder"]["inputs"]
    assert decoder["outputs"] == original["model"]["decoder"]["outputs"]
    assert decoder["filename"] == "model_ctx.onnx"
    assert decoder["session_options"] == {
        "intra_op_num_threads": 4,
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "provider_options": [{"OpenVINO": {"device_type": "NPU", "enable_causallm": "True"}}],
    }
    assert updated["model"]["vision"] == overrides["model"]["vision"]
    assert updated["search"] == overrides["search"]
    assert result.model_attributes["ort_genai_config_updates"] == [
        {"file_name": "genai_config.json", "json_paths": ["/model/decoder", "/model/vision", "/search"]}
    ]
    assert result.model_attributes["additional_files"] == [str(output_config)]
    assert source_config.read_bytes() == source_bytes
    assert not source_config.samefile(output_config)
    assert overrides == original_overrides


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "exactly one genai_config.json"),
        ("duplicate", "exactly one genai_config.json"),
        ("no_decoder", "does not define a decoder"),
    ],
)
def test_update_genai_config_rejects_invalid_source_config(tmp_path, case, message):
    source_config = tmp_path / "genai_config.json"
    source_config.write_text('{"model": {}}', encoding="utf-8")
    output_path = tmp_path / "model.onnx"
    output_path.touch()
    input_model = MagicMock()
    input_model.model_attributes = {
        "additional_files": [] if case == "missing" else [str(source_config)] * (2 if case == "duplicate" else 1)
    }
    output_model = ONNXModelHandler(model_path=str(output_path))
    encapsulation = create_pass_from_dict(OpenVINOEncapsulation, {"update_genai_config": True}, disable_search=True)

    with (
        patch.object(encapsulation, "_run_single_target", return_value=output_model),
        pytest.raises(ValueError, match=message),
    ):
        encapsulation.run(input_model, str(tmp_path))


def test_apply_genai_overrides_ignores_new_keys_by_default():
    defaults = {"model": {"decoder": {"filename": "old.onnx"}}}
    apply_genai_overrides(
        defaults,
        {"model": {"decoder": {"filename": "new.onnx", "new_key": True}}, "search": {"max_length": 32}},
    )

    assert defaults == {"model": {"decoder": {"filename": "new.onnx"}}}
