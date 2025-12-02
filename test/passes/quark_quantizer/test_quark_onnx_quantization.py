# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.quantization.calibrate import CalibrationDataReader

from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.quark_quantizer.quark_quantization import QuarkQuantization
from test.utils import get_onnx_model, get_pytorch_model_dummy_input


class DummyCalibrationDataReader(CalibrationDataReader):
    # pylint: disable=W0223
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.sample_counter = 500

    def get_next(self) -> dict:
        if self.sample_counter <= 0:
            return None

        data = get_pytorch_model_dummy_input()
        try:
            item = {"input": data}
            self.sample_counter -= 1
            return item
        except Exception:
            return None


@Registry.register_dataloader()
def _test_quant_dataloader(dataset, batch_size, **kwargs):
    return DummyCalibrationDataReader(batch_size=batch_size)


def test_static_qdq_u8s8_quantization(tmp_path):
    input_model = get_onnx_model()
    config = {
        "quant_mode": "static",
        "quant_format": "QDQ",
        "global_config": {
            "activation": {
                "symmetric": False,
                "calibration_method": "MinMax",
                "quant_granularity": "Tensor",
                "data_type": "UInt8",
            },
            "weight": {
                "symmetric": True,
                "calibration_method": "MinMax",
                "quant_granularity": "Tensor",
                "data_type": "Int8",
            },
        },
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
    }
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out is not None


def test_static_qdq_u8s8_with_mp_quantization(tmp_path):
    input_model = get_onnx_model()
    config = {
        "quant_mode": "static",
        "quant_format": "QDQ",
        "global_config": {
            "activation": {
                "symmetric": False,
                "calibration_method": "MinMax",
                "quant_granularity": "Tensor",
                "data_type": "UInt8",
            },
            "weight": {
                "symmetric": True,
                "calibration_method": "MinMax",
                "quant_granularity": "Tensor",
                "data_type": "Int8",
            },
        },
        "extra_options": {
            "MixedPrecisionTensor": {"UInt16": ["/fc1/Gemm_output_0"]},
        },
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
    }
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out is not None
