# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import patch

import numpy as np
import pytest
from onnxruntime.quantization.calibrate import CalibrationDataReader

from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.quark_quantizer.quark_quantization import QuarkQuantization
from test.utils import get_onnx_model, get_pytorch_model_dummy_input

pytestmark = pytest.mark.amd


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


def test_static_qdq_u8s8_with_tensorwise_mixed_precision_quantization(tmp_path):
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
            "TensorQuantOverrides": {"linear": [{"quant_type": "UInt16"}]},
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


def test_static_qdq_u8s8_with_opwise_mixed_precision_quantization(tmp_path):
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
        "layer_type_config": [
            [
                {
                    "activation": {
                        "data_type": "UInt16",
                        "symmetric": False,
                    },
                    "weight": {
                        "data_type": "UInt8",
                        "symmetric": False,
                    },
                },
                ["Gemm"],
            ]
        ],
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
    }
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out is not None


def test_static_qdq_u8s8_with_layerwise_mixed_precision_quantization(tmp_path):
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
        "specific_layer_config": [
            [
                {
                    "input_tensors": {
                        "data_type": "UInt16",
                        "symmetric": False,
                    },
                    "output_tensors": {
                        "data_type": "Int8",
                        "symmetric": True,
                    },
                    "weight": {
                        "data_type": "UInt8",
                        "symmetric": False,
                    },
                },
                ["node_linear"],
            ]
        ],
        "data_config": DataConfig(
            name="test_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader"),
        ),
    }
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    out = p.run(input_model, tmp_path)
    assert out is not None


@Registry.register_dataloader()
def _test_quant_dataloader_with_label(dataset, batch_size, **kwargs):
    """Yield model input and a label column as a real dataset pipeline would.

    The label must be filtered out before being passed to the ONNX calibration reader.
    """

    class _ReaderWithLabel(CalibrationDataReader):
        # pylint: disable=W0223
        def __init__(self):
            super().__init__()
            self.samples = [{"input": np.random.randn(1, 1).astype(np.float32), "label": 0} for _ in range(4)]
            self._iter = iter(self.samples)

        def get_next(self):
            return next(self._iter, None)

    return _ReaderWithLabel()


def test_calibration_dataloader_filters_label_columns(tmp_path):
    """Regression: Quark 0.12 rejects calibration inputs that are not model inputs.

    The pass must pass model_path/io_config to create_calibration_dataloader() so
    non-input columns (e.g. a label column) are stripped before reaching Quark.
    """
    input_model = get_onnx_model()
    config = {
        "quant_mode": "static",
        "quant_format": "QDQ",
        "global_config": {
            "activation": {"symmetric": False, "calibration_method": "MinMax", "data_type": "UInt8"},
            "weight": {"symmetric": True, "calibration_method": "MinMax", "data_type": "Int8"},
        },
        "data_config": DataConfig(
            name="test_quant_dc_config_with_label",
            load_dataset_config=DataComponentConfig(type="simple_dataset"),
            dataloader_config=DataComponentConfig(type="_test_quant_dataloader_with_label"),
        ),
    }
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    # Should complete without "Invalid input name: label" from Quark
    out = p.run(input_model, tmp_path)
    assert out is not None


def test_onnx_path_raises_clear_error_for_old_quark(tmp_path):
    """ONNX path must raise a clear ValueError, not an ImportError, for amd-quark < 0.12.0."""
    input_model = get_onnx_model()
    config = {"quant_mode": "static", "quant_format": "QDQ"}
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    with patch("quark.__version__", "0.11.2"), pytest.raises(ValueError, match=r"amd-quark>=0\.12\.0"):
        p.run(input_model, tmp_path)


def test_torch_path_raises_clear_error_for_old_quark(tmp_path):
    """Torch path must raise a clear ValueError, not an ImportError, for amd-quark < 0.12.0."""
    from unittest.mock import MagicMock

    from olive.model import HfModelHandler

    # Use a MagicMock so handler construction/validation is bypassed.
    # The version gate fires inside _run_quark_torch before any model loading.
    input_model = MagicMock(spec=HfModelHandler)
    config = {"quant_scheme": "uint4_wo_128"}
    p = create_pass_from_dict(QuarkQuantization, config, disable_search=True)
    with patch("quark.__version__", "0.11.2"), pytest.raises(ValueError, match=r"amd-quark>=0\.12\.0"):
        p.run(input_model, tmp_path)
