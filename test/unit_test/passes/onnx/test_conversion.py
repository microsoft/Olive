# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from test.unit_test.utils import get_hf_model_with_past, get_onnx_model, get_pytorch_model
from unittest.mock import patch

import pytest

from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion, OnnxOpVersionConversion


@pytest.mark.parametrize("input_model", [get_pytorch_model(), get_hf_model_with_past()])
def test_onnx_conversion_pass(input_model, tmp_path):
    # setup
    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # The conversion need torch version > 1.13.1, otherwise, it will complain
    # Unsupported ONNX opset version: 18
    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()


# TODO(jambayk): add case for when quantized_modules is not provided when gpu test is added
# patch needed since bitsandbytes is not installed in the test environment
@patch("olive.model.config.hf_config.HfFromPretrainedArgs.get_quantization_config")
def test_onnx_conversion_pass_quant_model(_, tmp_path):
    # setup
    quantized_modules = ["v_proj", "k_proj", "fc_in", "fc_out", "out_proj", "q_proj"]
    input_model = PyTorchModelHandler(
        hf_config={
            "model_name": "hf-internal-testing/tiny-random-gptj",
            "task": "text-generation",
            "from_pretrained_args": {
                "quantization_method": "bitsandbytes",
                "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
            },
        },
        model_attributes={"quantized_modules": quantized_modules},
    )
    p = create_pass_from_dict(OnnxConversion, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()
    assert set(onnx_model.model_attributes["quantized_modules"]) == set(quantized_modules)
    assert onnx_model.model_attributes["quantization_config"]["load_in_4bit"] is True
    assert onnx_model.model_attributes["quantization_config"]["bnb_4bit_quant_type"] == "nf4"


@pytest.mark.parametrize("target_opset", [9, 10, 16])
def test_onnx_op_version_conversion_pass(target_opset, tmp_path):
    input_model = get_onnx_model()
    # setup
    p = create_pass_from_dict(OnnxOpVersionConversion, {"target_opset": target_opset}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    onnx_model = p.run(input_model, None, output_folder)

    # assert
    assert onnx_model.load_model().opset_import[0].version == target_opset
