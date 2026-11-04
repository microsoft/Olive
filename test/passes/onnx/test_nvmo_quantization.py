# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import onnxruntime
import pytest
import torch
from packaging import version
from transformers import GPT2Config

from olive.constants import Precision
from olive.data.config import DataConfig
from olive.model import HfModelHandler
from olive.package_config import OlivePackageConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.nvmo_quantization import NVModelOptQuantization
from test.utils import get_onnx_model


@pytest.fixture(name="nvfp4_pass")
def fixture_nvfp4_pass():
    return create_pass_from_dict(
        NVModelOptQuantization,
        {"precision": "nvfp4", "data_config": DataConfig(name="calibration")},
        disable_search=True,
    )


@pytest.fixture(name="mock_modelopt")
def fixture_mock_modelopt(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in (
            "modelopt",
            "modelopt.torch",
            "modelopt.torch.quantization",
            "modelopt.torch.export",
            "modelopt.onnx",
            "modelopt.onnx.quantization",
            "modelopt.onnx.quantization.int4",
        )
    }
    quantization = modules["modelopt.torch.quantization"]
    quantization.NVFP4_DEFAULT_CFG = {"quant_cfg": {"*weight_quantizer": {"num_bits": (2, 1)}}, "algorithm": "max"}
    quantization.quantize = Mock()
    modules["modelopt.torch.export"].export_hf_checkpoint = Mock()
    modules["modelopt.onnx.quantization.int4"].quantize = Mock()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def test_nvmo_config_accepts_nvfp4(nvfp4_pass):
    assert NVModelOptQuantization.validate_config(nvfp4_pass.config, nvfp4_pass.accelerator_spec)
    package_config = OlivePackageConfig.load_default_config()
    assert Precision.NVFP4 in package_config.get_pass_module_config("NVModelOptQuantization").supported_precisions


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_config": None},
        {"use_random_calibration_data": True},
        {"algorithm": "rtn"},
        {"calibration_method": "rtn"},
        {"calibration_providers": ["cuda"]},
        {"tokenizer_dir": "some-tokenizer"},
        {"calibration_params": {"calib_size": 8}},
        {"int4_block_size": 32},
        {"nodes_to_exclude": []},
        {"enable_mixed_quant": True},
        {"layers_8bit": "lm_head"},
        {"trust_remote_code": True},
    ],
)
def test_nvmo_config_rejects_int4_options_for_nvfp4(nvfp4_pass, overrides, monkeypatch):
    log_error = Mock()
    monkeypatch.setattr("olive.passes.onnx.nvmo_quantization.logger.error", log_error)
    config = nvfp4_pass.config.model_copy(update=overrides)
    assert not NVModelOptQuantization.validate_config(config, nvfp4_pass.accelerator_spec)
    log_error.assert_called_once()


@pytest.mark.parametrize("precision", ["int8", "fp8"])
def test_nvmo_config_rejects_unsupported_precision(precision):
    p = create_pass_from_dict(
        NVModelOptQuantization, {"precision": precision, "use_random_calibration_data": True}, disable_search=True
    )
    assert not NVModelOptQuantization.validate_config(p.config, p.accelerator_spec)


def test_nvmo_config_preserves_int4_defaults():
    p = create_pass_from_dict(NVModelOptQuantization, {"use_random_calibration_data": True}, disable_search=True)
    assert p.config.precision == Precision.INT4
    assert p.config.calibration_method == NVModelOptQuantization.CalibrationMethod.AWQ_LITE
    assert NVModelOptQuantization.validate_config(p.config, p.accelerator_spec)
    config = p.config.model_copy(update={"data_config": DataConfig(name="calibration")})
    assert not NVModelOptQuantization.validate_config(config, p.accelerator_spec)


def test_nvmo_nvfp4_rejects_onnx_input(nvfp4_pass, tmp_path):
    with pytest.raises(ValueError, match="NVFP4 requires an HfModel input"):
        nvfp4_pass.run(get_onnx_model(), str(tmp_path / "output"))


def test_nvmo_nvfp4_reports_missing_dependency(nvfp4_pass, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "modelopt.torch.export", None)
    with pytest.raises(ImportError, match="nvidia-modelopt"):
        nvfp4_pass.quantize_nvfp4(Mock(spec=HfModelHandler), nvfp4_pass.config, str(tmp_path))


def test_nvmo_nvfp4_requires_cuda(nvfp4_pass, mock_modelopt, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        nvfp4_pass.quantize_nvfp4(Mock(spec=HfModelHandler), nvfp4_pass.config, str(tmp_path))


def test_nvmo_nvfp4_rejects_unmerged_adapters(nvfp4_pass, mock_modelopt, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = Mock(spec=HfModelHandler)
    model.adapter_path = "adapter"
    with pytest.raises(ValueError, match="Merge"):
        nvfp4_pass.run(model, str(tmp_path))


@pytest.mark.parametrize("device_map", [None, "cuda"])
def test_nvmo_nvfp4_calibrates_and_exports_hf_checkpoint(nvfp4_pass, mock_modelopt, monkeypatch, tmp_path, device_map):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    input_path = tmp_path / "input"
    hf_config = GPT2Config()
    hf_config.save_pretrained(input_path)
    model = HfModelHandler(input_path, task="text-generation", load_kwargs={"device_map": device_map})
    torch_model = Mock(spec=torch.nn.Module)
    torch_model.eval.return_value = torch_model
    torch_model.to.return_value = torch_model
    torch_model.parameters.return_value = iter([torch.nn.Parameter(torch.ones(1))])
    model.load_model = Mock(return_value=torch_model)
    model.save_metadata = Mock()
    batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.int64)}
    get_dataset = Mock(return_value=[batch])
    monkeypatch.setattr("olive.passes.onnx.nvmo_quantization.get_calibration_dataset", get_dataset)
    quantization = mock_modelopt["modelopt.torch.quantization"]

    def quantize(calibration_model, quant_config, forward_loop):
        assert not torch.is_grad_enabled()
        assert quant_config == quantization.NVFP4_DEFAULT_CFG
        assert quant_config is not quantization.NVFP4_DEFAULT_CFG
        quant_config["algorithm"] = "modified-by-backend"
        forward_loop(calibration_model)
        return calibration_model

    quantization.quantize.side_effect = quantize
    exporter = mock_modelopt["modelopt.torch.export"].export_hf_checkpoint

    def export_checkpoint(quantized_model, export_dir):
        assert quantized_model is torch_model
        hf_config.quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        hf_config.save_pretrained(export_dir)

    exporter.side_effect = export_checkpoint
    output_path = str(tmp_path / "output")
    output_model = nvfp4_pass.run(model, output_path)

    assert isinstance(output_model, HfModelHandler)
    assert output_model.model_path == output_path
    assert output_model.get_hf_model_config().quantization_config["quant_algo"] == "NVFP4"
    get_dataset.assert_called_once_with(model, nvfp4_pass.config.data_config)
    model.load_model.assert_called_once_with(cache_model=False)
    if device_map:
        torch_model.to.assert_not_called()
    else:
        torch_model.to.assert_called_once_with("cuda")
    torch_model.assert_called_once_with(**batch, use_cache=False)
    assert quantization.NVFP4_DEFAULT_CFG["algorithm"] == "max"
    exporter.assert_called_once_with(torch_model, export_dir=output_path)
    model.save_metadata.assert_called_once_with(output_path)
    mock_modelopt["modelopt.onnx.quantization.int4"].quantize.assert_not_called()


def test_nvmo_int4_keeps_modelopt_options_and_onnx_output(mock_modelopt, tmp_path):
    model = get_onnx_model()
    quantize_int4 = mock_modelopt["modelopt.onnx.quantization.int4"].quantize
    quantize_int4.return_value = model.load_model()
    p = create_pass_from_dict(
        NVModelOptQuantization,
        {
            "use_random_calibration_data": True,
            "calibration_providers": ["cpu"],
            "int4_block_size": 32,
            "enable_mixed_quant": True,
            "layers_8bit": "layers.0",
        },
        disable_search=True,
    )
    output = p.run(model, str(tmp_path / "output"))
    assert Path(output.model_path).is_file()
    assert next(op.version for op in output.load_model().opset_import if op.domain == "") >= 21
    quantize_int4.assert_called_once_with(
        model.model_path,
        calibration_method="awq_lite",
        calibration_data_reader=None,
        calibration_eps=["cpu"],
        block_size=32,
        nodes_to_exclude=["/lm_head"],
        input_shapes_profile=None,
        enable_mixed_quant=True,
        layers_8bit="layers.0",
    )
    mock_modelopt["modelopt.torch.quantization"].quantize.assert_not_called()


def test_nvmo_native_nvfp4_exports_packed_qwen_checkpoint_on_cpu(tmp_path):
    pytest.importorskip("modelopt", minversion="0.48.0")
    from modelopt.torch.export import export_hf_checkpoint
    from modelopt.torch.quantization import NVFP4_DEFAULT_CFG, quantize
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, Qwen3_5Config

    text_config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
    }
    model_config = Qwen3_5Config.from_dict({"text_config": text_config})
    model = AutoModelForCausalLM.from_config(model_config).eval().cpu()
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.int64)}
    with torch.no_grad():
        quantized = quantize(model, NVFP4_DEFAULT_CFG, forward_loop=lambda m: m(**batch, use_cache=False))
        export_hf_checkpoint(quantized, export_dir=tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "qwen3_5_text"
    assert config["quantization_config"]["quant_algo"] == "NVFP4"
    assert config["quantization_config"]["group_size"] == 16
    assert (tmp_path / "hf_quant_config.json").is_file()
    with safe_open(tmp_path / "model.safetensors", framework="pt") as checkpoint:
        weights = [
            checkpoint.get_tensor(name) for name in sorted(checkpoint.keys()) if name.endswith(".mlp.up_proj.weight")
        ]
    assert len(weights) == 2
    assert all(weight.dtype == torch.uint8 and weight.numel() == 32 * 64 // 2 for weight in weights)


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
        "calibration_method": "awq_lite",
        "use_random_calibration_data": True,
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


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) > version.parse("1.20.1"),
    reason="ORT 1.21 doesn't support Volta anymore. Reenable this test once we switch to a new SKU.",
)
def test_nvmo_quantization_with_mixed_precision(tmp_path):
    """Test NVModelOptQuantization with mixed precision enabled."""
    ov_model = get_onnx_model()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Configuration with mixed precision enabled
    config = {
        "calibration_method": "awq_lite",
        "use_random_calibration_data": True,
        "enable_mixed_quant": True,
    }

    output_folder = str(tmp_path / "quantized_mixed")

    # Create NVModelOptQuantization pass and run quantization
    p = create_pass_from_dict(NVModelOptQuantization, config, disable_search=True)
    quantized_model = p.run(ov_model, output_folder)

    # Assertions to check if quantization was successful
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "DequantizeLinear" in [node.op_type for node in quantized_model.load_model().graph.node]


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) > version.parse("1.20.1"),
    reason="ORT 1.21 doesn't support Volta anymore. Reenable this test once we switch to a new SKU.",
)
def test_nvmo_quantization_with_layers_8bit(tmp_path):
    """Test NVModelOptQuantization with layers_8bit parameter."""
    ov_model = get_onnx_model()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Configuration with layers_8bit specified
    config = {
        "calibration_method": "awq_lite",
        "use_random_calibration_data": True,
        "layers_8bit": "layers.0,lm_head",
    }

    output_folder = str(tmp_path / "quantized_layers_8bit")

    # Create NVModelOptQuantization pass and run quantization
    p = create_pass_from_dict(NVModelOptQuantization, config, disable_search=True)
    quantized_model = p.run(ov_model, output_folder)

    # Assertions to check if quantization was successful
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "DequantizeLinear" in [node.op_type for node in quantized_model.load_model().graph.node]
