# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from test.unit_test.utils import get_onnx_model

import numpy as np
import pytest
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version

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


def dummy_calibration_reader(data_dir=None, batch_size=1, *args, **kwargs):
    return RandomDataReader()


@pytest.mark.skipif(
    version.parse(OrtVersion) >= version.parse("1.16.0"),
    reason="VitisAIQuantization is not supported in ORT 1.16.0 with TensorsData",
)
def test_vitis_ai_quantization_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    dummy_user_script = str(tmp_path / "dummy_user_script.py")
    dummy_data = str(tmp_path / "dummy_data")
    with open(dummy_user_script, "w") as f:
        f.write(" ")
    if not os.path.exists(dummy_data):
        os.mkdir(dummy_data)

    config = {"user_script": dummy_user_script, "data_dir": dummy_data, "dataloader_func": dummy_calibration_reader}
    output_folder = str(tmp_path / "vitis_ai_quantized")

    # create VitisAIQuantization pass
    p = create_pass_from_dict(VitisAIQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(input_model, None, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
