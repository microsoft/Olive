# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model

import numpy as np
import pytest
from onnxruntime.quantization.calibrate import CalibrationDataReader

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.vitis_ai_quantization import VitisAIQuantization


class RandomDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data_dicts = []
        self.flag = True

    def get_next(self):
        if self.flag:
            self.flag = False
            input_shape = [1, 1]
            data = []
            dummy_data = np.random.random(input_shape).astype(np.float32)
            data.append({"input": dummy_data})

            self.enum_data_dicts = iter(data)
        return next(self.enum_data_dicts, None)


def dummy_calibration_reader(data_dir, batch_size, *args, **kwargs):
    return RandomDataReader()


@pytest.mark.parametrize("calibrate_method", ["MinMSE", "NonOverflow"])
def test_vitis_ai_quantization_pass(calibrate_method, tmp_path):
    # setup
    input_model = get_onnx_model()
    dummy_user_script = tmp_path / "dummy_user_script.py"
    dummy_data: Path = tmp_path / "dummy_data"
    with dummy_user_script.open("w") as f:
        f.write(" ")
    if not dummy_data.exists():
        dummy_data.mkdir()

    config = {
        "user_script": str(dummy_user_script),
        "data_dir": str(dummy_data),
        "dataloader_func": dummy_calibration_reader,
        "calibrate_method": calibrate_method,
    }
    output_folder = str(tmp_path / "vitis_ai_quantized")

    # create VitisAIQuantization pass
    p = create_pass_from_dict(VitisAIQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(input_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
