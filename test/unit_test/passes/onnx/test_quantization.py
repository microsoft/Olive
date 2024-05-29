from test.unit_test.utils import get_onnx_model, get_pytorch_model_dummy_input

import pytest
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version

from olive.common.pydantic_v1 import ValidationError
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.quantization import OnnxMatMul4Quantizer, OnnxQuantization, OnnxStaticQuantization


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
def test_static_quantization(calibrate_method, tmp_path):
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


def test_dynamic_quantization(tmp_path):
    input_model = get_onnx_model()
    config = {"quant_mode": "dynamic"}
    p = create_pass_from_dict(OnnxQuantization, config, disable_search=True)

    ort_version = version.parse(OrtVersion)
    if ort_version.major == 1 and ort_version.minor == 17:
        # there is a bug in ort quantizer in versions 1.17.x
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'initial_type'"):
            _ = p.run(input_model, None, tmp_path)
    else:
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
        "WeightSymmetric": None,
        "ActivationSymmetric": True,
        "qnn_extra_options": {
            "init_overrides": None,
            "add_qtype_converts": True,
        },
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="NPU",
        execution_provider="QNNExecutionProvider",
    )
    p = create_pass_from_dict(OnnxStaticQuantization, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, None, tmp_path)
    assert out is not None


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.16.2"),
    reason="matmul 4bit quantization is only supported in onnxruntime>=1.16.2",
)
@pytest.mark.parametrize(
    ("algorithm", "weight_only_quant_configs"),
    [
        (None, None),
        ("RTN", {"ratios": {}}),
    ],
)
def test_matmul_4bit_quantization_without_dataloader(tmp_path, algorithm, weight_only_quant_configs):
    input_model = get_onnx_model()
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 4,
        "algorithm": algorithm,
        "weight_only_quant_configs": weight_only_quant_configs,
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    p = create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, None, tmp_path)
    assert out is not None


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.18.0"),
    reason="matmul 4bit quantization with `DEFAULT` and `HQQ` is only supported in onnxruntime<1.18.0",
)
@pytest.mark.parametrize(
    ("algorithm", "weight_only_quant_configs"),
    [
        ("DEFAULT", None),
        ("HQQ", None),
    ],
)
def test_matmul_4bit_quantization_without_dataloader_ort_1_18(tmp_path, algorithm, weight_only_quant_configs):
    input_model = get_onnx_model()
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 4,
        "algorithm": algorithm,
        "weight_only_quant_configs": weight_only_quant_configs,
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    p = create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, None, tmp_path)
    assert out is not None


def test_matmul_gptq_with_dataloader(tmp_path):
    input_model = get_onnx_model()
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 4,
        "algorithm": "GPTQ",
        "dataloader_func": dummpy_dataloader_func,
        "weight_only_quant_configs": {"percdamp": 0.01, "block_size": 128, "use_less_config": 1},
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    p = create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
    out = p.run(input_model, None, tmp_path)
    assert out is not None


@pytest.mark.skipif(
    version.parse(OrtVersion) < version.parse("1.16.2"),
    reason="matmul 4bit quantization is only supported in onnxruntime>=1.16.2",
)
def test_invalid_config_for_matmul_4bits():
    config = {
        "block_size": 32,
        "is_symmetric": True,
        "nodes_to_exclude": [],
        "accuracy_level": 5,
        "algorithm": "TE",
    }
    accelerator_spec = AcceleratorSpec(
        accelerator_type="CPU",
        execution_provider="CPUExecutionProvider",
    )
    with pytest.raises(ValidationError):
        create_pass_from_dict(OnnxMatMul4Quantizer, config, disable_search=True, accelerator_spec=accelerator_spec)
