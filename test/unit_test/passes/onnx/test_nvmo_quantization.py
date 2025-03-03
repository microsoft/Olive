# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model

import onnxruntime
import pytest
from packaging import version

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.nvmo_quantization import NVModelOptQuantization


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) > version.parse("1.20.1"),
    reason="ORT 1.21 doesn't support Volta anymore. Reenable this test once we switch to a new SKU.",
)
def test_nvmo_quantization(tmp_path):
    ov_model = get_onnx_model()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Configuration with default values and random_calib_data set to True
    config = {
        "calibration": "awq_lite",
        "random_calib_data": True,
    }

    output_folder = str(tmp_path / "quantized")

    # Create NVModelOptQuantization pass and run quantization
    p = create_pass_from_dict(NVModelOptQuantization, config, disable_search=True)
    quantized_model = p.run(ov_model, output_folder)

    # Assertions to check if quantization was successful
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "DequantizeLinear" in [node.op_type for node in quantized_model.load_model().graph.node]
