from test.unit_test.utils import get_onnx_model, get_pytorch_model_dummy_input

import pytest
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version

from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OnnxQuantization, OnnxStaticQuantization


class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.sample_counter = 500

    def get_next(self) -> dict:
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


@pytest.mark.parametrize("calibrate_method", ["MinMax", "Entropy", "Percentile"])
def test_quantization(calibrate_method, tmp_path):
    input_model = get_onnx_model()
    config = {
        "quant_mode": "static",
        "calibrate_method": calibrate_method,
        "quant_format": "QOperator",
        "MatMulConstBOnly": False,
        "per_channel": True,
        "reduce_range": True,
        "dataloader_func": dummpy_dataloader_func,
        "weight_type": "QUInt8",
        "activation_type": "QUInt8",
        "quant_preprocess": True,
    }
    p = create_pass_from_dict(OnnxQuantization, config, disable_search=True)
    out = p.run(input_model, None, tmp_path)
    assert out is not None


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.17.0"),
    reason="qnn quantization is only supported in onnxruntime>=1.17.0",
)
def test_qnn_quantization(tmp_path):
    input_model = get_onnx_model()
    config = {
        "quant_format": "QDQ",
        "dataloader_func": dummpy_dataloader_func,
        "weight_type": "QUInt8",
        "activation_type": "QUInt16",
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="NPU",
        execution_provider="QNNExecutionProvider",
    )
    p = create_pass_from_dict(OnnxStaticQuantization, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, None, tmp_path)
    assert out is not None
