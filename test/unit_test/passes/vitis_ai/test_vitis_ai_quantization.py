# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model

import onnxruntime
import pytest
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version

from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
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
            dummy_data = torch.rand(input_shape).to(torch.float32)
            data.append({"input": dummy_data})

            self.enum_data_dicts = iter(data)
        return next(self.enum_data_dicts, None)


@Registry.register_dataloader()
def dummy_calibration_reader(dataset, batch_size, **kwargs):
    return RandomDataReader()


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) >= version.parse("1.20"),
    reason="Fails on onnxruntime 1.20+",
)
@pytest.mark.parametrize("calibrate_method", ["MinMSE", "NonOverflow"])
def test_vitis_ai_quantization_pass(calibrate_method, tmp_path):
    # setup
    input_model = get_onnx_model()
    config = {
        "data_config": DataConfig(
            name="test_vitis_ai_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="dummy_calibration_reader"),
        ),
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
