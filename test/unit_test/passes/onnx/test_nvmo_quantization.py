# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model, get_pytorch_model_dummy_input
from typing import Any, Dict, Optional

from onnxruntime.quantization.calibrate import CalibrationDataReader  # type: ignore[import]

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.nvmo_quantization import NVModelOptQuantization


class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.sample_counter = 64

    def get_next(self) -> Optional[Dict[Any, Any]]:
        if self.sample_counter <= 0:
            return None

        data = get_pytorch_model_dummy_input()
        try:
            item = {"input": data.numpy()}
            self.sample_counter -= 1
            return item
        except Exception:
            return None


def dummpy_dataloader_func(data_dir, batch_size, *args, **kwargs):
    return DummyCalibrationDataReader(data_dir, batch_size=batch_size)


def test_nvmo_quantization(tmp_path):
    ov_model = get_onnx_model()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {"data_dir": data_dir, "dataloader_func": dummpy_dataloader_func}
    output_folder = str(tmp_path / "quantized")

    # create NVModelOptQuantization pass and run quantization
    p = create_pass_from_dict(NVModelOptQuantization, config, disable_search=True)
    quantized_model = p.run(ov_model, output_folder)

    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "DequantizeLinear" in [node.op_type for node in quantized_model.load_model().graph.node]
