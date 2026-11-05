# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GPT2Config

from olive.data.config import DataConfig
from olive.model import HfModelHandler
from olive.package_config import OlivePackageConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.modelopt import ModelOptQuantization
from test.utils import get_onnx_model


@pytest.fixture(name="mock_modelopt")
def fixture_mock_modelopt(monkeypatch):
    modules = {
        name: ModuleType(name)
        for name in (
            "modelopt",
            "modelopt.torch",
            "modelopt.torch.opt",
            "modelopt.torch.quantization",
            "modelopt.torch.quantization.config",
            "modelopt.torch.export",
            "modelopt.recipe",
            "modelopt.recipe.presets",
        )
    }
    presets = {
        name: {"quant_cfg": [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": bits}}], "algorithm": "max"}
        for name, bits in (
            ("fp8", (4, 3)),
            ("int8_smoothquant", 8),
            ("int4_awq", 4),
            ("nvfp4", (2, 1)),
            ("mxfp4", (2, 1)),
            ("mxfp6", (3, 2)),
            ("mxfp8", (4, 3)),
            ("mxint8", 8),
            ("future_precision", (5, 2)),
        )
    }
    modules["modelopt.recipe.presets"].QUANT_CFG_CHOICES = presets
    modules["modelopt.torch.quantization"].quantize = Mock()
    modules["modelopt.torch.quantization.config"].need_calibration = Mock(return_value=True)
    modules["modelopt.torch.export"].export_hf_checkpoint = Mock()
    modules["modelopt.torch.opt"].enable_huggingface_checkpointing = Mock()
    modules["modelopt"].torch = modules["modelopt.torch"]
    modules["modelopt.torch"].quantization = modules["modelopt.torch.quantization"]
    modules["modelopt.torch"].opt = modules["modelopt.torch.opt"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        presets=presets,
        quantize=modules["modelopt.torch.quantization"].quantize,
        need_calibration=modules["modelopt.torch.quantization.config"].need_calibration,
        export=modules["modelopt.torch.export"].export_hf_checkpoint,
        enable_checkpointing=modules["modelopt.torch.opt"].enable_huggingface_checkpointing,
    )


@pytest.fixture(name="hf_model")
def fixture_hf_model(tmp_path):
    path = tmp_path / "input"
    GPT2Config().save_pretrained(path)
    model = HfModelHandler(path, task="text-generation")
    torch_model = Mock(spec=torch.nn.Module)
    torch_model.eval.return_value = torch_model
    torch_model.to.return_value = torch_model
    torch_model.config = SimpleNamespace(use_cache=True)
    torch_model.get_input_embeddings = Mock(return_value=torch.nn.Embedding(8, 4))
    torch_model.save_pretrained = Mock()
    model.load_model = Mock(return_value=torch_model)
    model.save_metadata = Mock()
    return model


def _create_pass(**config):
    return create_pass_from_dict(ModelOptQuantization, {"device": "cpu", **config}, disable_search=True)


def test_modelopt_registration_and_defaults():
    package_config = OlivePackageConfig.load_default_config()
    pinfo = package_config.get_pass_module_config("ModelOptQuantization")
    assert pinfo.module_path == "olive.passes.pytorch.modelopt.ModelOptQuantization"
    assert {"nvfp4", "mxfp4", "mxfp6", "mxfp8", "mxint8", "fp8", "int4", "int8"} <= pinfo.supported_precisions
    p = _create_pass()
    assert p.config.quantization_config == "fp8"
    assert p.config.export_format == "hf"
    assert p.validate_config(p.config, p.accelerator_spec)


@pytest.mark.parametrize(
    "config",
    [
        {"quantization_config": ""},
        {"quantization_config": {}},
        {"quantization_config": {"algorithm": "max"}},
        {"device": "invalid-device"},
        {"device": "meta"},
    ],
)
def test_modelopt_config_rejects_invalid_values(config, monkeypatch):
    error = Mock()
    monkeypatch.setattr("olive.passes.pytorch.modelopt.logger.error", error)
    monkeypatch.setattr("olive.passes.pytorch.modelopt.logger.exception", error)
    p = _create_pass(**config)
    assert not p.validate_config(p.config, p.accelerator_spec)
    error.assert_called_once()


@pytest.mark.parametrize(
    "preset",
    ["fp8", "int8_smoothquant", "int4_awq", "nvfp4", "mxfp4", "mxfp6", "mxfp8", "mxint8", "future_precision"],
)
@pytest.mark.parametrize("export_format", ["hf", "modelopt"])
def test_modelopt_quantizes_native_presets_and_saves_hf_model(
    preset, export_format, mock_modelopt, hf_model, monkeypatch, tmp_path
):
    batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.int64)}
    data_config = DataConfig(name="calibration")
    get_dataset = Mock(return_value=[batch])
    monkeypatch.setattr("olive.passes.pytorch.modelopt.get_calibration_dataset", get_dataset)
    torch_model = hf_model.load_model.return_value
    original_config = deepcopy(mock_modelopt.presets[preset])

    def quantize(model, config, forward_loop):
        assert model is torch_model
        assert not torch.is_grad_enabled()
        assert model.config.use_cache is False
        assert config == original_config
        config["algorithm"] = "modified-by-backend"
        forward_loop(model)
        return model

    def save_config(*args, **kwargs):
        path = kwargs["export_dir"] if kwargs else args[0]
        config = GPT2Config()
        config.quantization_config = {"quant_method": "modelopt", "quant_algo": preset}
        config.save_pretrained(path)

    mock_modelopt.quantize.side_effect = quantize
    mock_modelopt.export.side_effect = save_config
    torch_model.save_pretrained.side_effect = lambda path, **kwargs: save_config(path)
    p = _create_pass(quantization_config=preset, data_config=data_config, export_format=export_format)
    output_path = str(tmp_path / "output")
    result = p.run(hf_model, output_path)

    assert isinstance(result, HfModelHandler)
    assert result.model_path == output_path
    assert result.get_hf_model_config().quantization_config["quant_algo"] == preset
    assert mock_modelopt.presets[preset] == original_config
    hf_model.load_model.assert_called_once_with(cache_model=False)
    torch_model.eval.assert_called_once()
    torch_model.to.assert_called_once_with(torch.device("cpu"))
    torch_model.assert_called_once_with(**batch)
    assert torch_model.config.use_cache is True
    hf_model.save_metadata.assert_called_once_with(output_path)
    get_dataset.assert_called_once_with(hf_model, p.config.data_config)
    if export_format == "hf":
        mock_modelopt.export.assert_called_once_with(torch_model, export_dir=output_path)
        torch_model.save_pretrained.assert_not_called()
        mock_modelopt.enable_checkpointing.assert_not_called()
    else:
        mock_modelopt.export.assert_not_called()
        mock_modelopt.enable_checkpointing.assert_called_once()
        torch_model.save_pretrained.assert_called_once_with(output_path, save_modelopt_state=True)


@pytest.mark.parametrize("device_map", ["cpu", {"": "cpu"}, 0])
def test_modelopt_preserves_device_map_and_skips_unneeded_calibration(
    device_map, mock_modelopt, hf_model, monkeypatch, tmp_path
):
    hf_model.get_load_kwargs = Mock(return_value={"device_map": device_map})
    torch_model = hf_model.load_model.return_value
    mock_modelopt.need_calibration.return_value = False
    mock_modelopt.quantize.return_value = torch_model
    get_dataset = Mock()
    monkeypatch.setattr("olive.passes.pytorch.modelopt.get_calibration_dataset", get_dataset)
    inherit = Mock(return_value=hf_model)
    monkeypatch.setattr("olive.passes.pytorch.modelopt.inherit_hf_from_hf", inherit)
    _create_pass(device="cuda").run(hf_model, str(tmp_path / "output"))
    torch_model.to.assert_not_called()
    get_dataset.assert_not_called()
    assert mock_modelopt.quantize.call_args.kwargs["forward_loop"] is None


def test_modelopt_accepts_custom_native_config(mock_modelopt, hf_model, monkeypatch, tmp_path):
    config = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None}
    mock_modelopt.need_calibration.return_value = False
    mock_modelopt.quantize.return_value = hf_model.load_model.return_value
    monkeypatch.setattr("olive.passes.pytorch.modelopt.inherit_hf_from_hf", Mock(return_value=hf_model))
    _create_pass(quantization_config=config).run(hf_model, str(tmp_path / "output"))
    resolved = mock_modelopt.quantize.call_args.args[1]
    assert resolved == config
    assert resolved is not config


def test_modelopt_calibrates_model_without_generation_cache(mock_modelopt, hf_model, monkeypatch, tmp_path):
    torch_model = hf_model.load_model.return_value
    torch_model.config = SimpleNamespace()
    batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.int64)}
    monkeypatch.setattr("olive.passes.pytorch.modelopt.get_calibration_dataset", Mock(return_value=[batch]))
    monkeypatch.setattr("olive.passes.pytorch.modelopt.inherit_hf_from_hf", Mock(return_value=hf_model))

    def quantize(model, config, forward_loop):
        forward_loop(model)
        return model

    mock_modelopt.quantize.side_effect = quantize
    _create_pass().run(hf_model, str(tmp_path / "output"))
    torch_model.assert_called_once_with(**batch)
    assert not hasattr(torch_model.config, "use_cache")


def test_modelopt_rejects_unknown_preset_before_loading_model(mock_modelopt, hf_model, tmp_path):
    with pytest.raises(ValueError, match="Available presets:"):
        _create_pass(quantization_config="unknown").run(hf_model, str(tmp_path))
    hf_model.load_model.assert_not_called()


def test_modelopt_rejects_onnx_input(tmp_path):
    with pytest.raises(ValueError, match="requires an HfModel"):
        _create_pass().run(get_onnx_model(), str(tmp_path))


def test_modelopt_rejects_unmerged_adapters(hf_model, tmp_path):
    hf_model.set_resource("adapter_path", "adapter")
    with pytest.raises(ValueError, match="Merge"):
        _create_pass().run(hf_model, str(tmp_path))


def test_modelopt_reports_missing_dependency(hf_model, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "modelopt.torch.quantization", None)
    with pytest.raises(ImportError, match="nvidia-modelopt"):
        _create_pass().run(hf_model, str(tmp_path))


def test_modelopt_reports_unavailable_cuda(mock_modelopt, hf_model, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        _create_pass(device="cuda").run(hf_model, str(tmp_path))
    hf_model.load_model.assert_not_called()


def test_modelopt_reports_offloaded_input_embeddings(mock_modelopt, hf_model, monkeypatch, tmp_path):
    hf_model.load_model.return_value.get_input_embeddings.return_value = torch.nn.Embedding(8, 4, device="meta")
    monkeypatch.setattr("olive.passes.pytorch.modelopt.get_calibration_dataset", Mock(return_value=[{}]))
    with pytest.raises(ValueError, match="materialized input embeddings"):
        _create_pass().run(hf_model, str(tmp_path / "output"))
    mock_modelopt.quantize.assert_not_called()


@pytest.mark.parametrize("stage", ["quantize", "export"])
def test_modelopt_propagates_backend_errors(stage, mock_modelopt, hf_model, monkeypatch, tmp_path):
    mock_modelopt.need_calibration.return_value = False
    mock_modelopt.quantize.return_value = hf_model.load_model.return_value
    getattr(mock_modelopt, stage).side_effect = NotImplementedError("format is unsupported")
    with pytest.raises(NotImplementedError, match="format is unsupported"):
        _create_pass().run(hf_model, str(tmp_path / "output"))
    hf_model.save_metadata.assert_not_called()
    assert hf_model.load_model.return_value.config.use_cache is True


@pytest.mark.parametrize("has_state", [False, True])
@pytest.mark.parametrize("subfolder", ["", "weights"])
def test_hf_model_restores_modelopt_state_only_when_present(mock_modelopt, tmp_path, monkeypatch, has_state, subfolder):
    path = tmp_path / "checkpoint"
    GPT2Config().save_pretrained(path / subfolder)
    if has_state:
        (path / subfolder / "modelopt_state.pth").touch()
    model = HfModelHandler(path, task="text-generation", load_kwargs={"subfolder": subfolder})
    load = Mock(return_value=torch.nn.Linear(4, 4))

    def load_model(*args, **kwargs):
        if has_state:
            mock_modelopt.enable_checkpointing.assert_called_once()
        else:
            mock_modelopt.enable_checkpointing.assert_not_called()
        return load(*args, **kwargs)

    monkeypatch.setattr("olive.model.handler.hf.load_model_from_task", load_model)
    assert model.load_model() is load.return_value
    assert model.load_model() is load.return_value
    load.assert_called_once()


def test_hf_model_reports_missing_modelopt_for_state_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    GPT2Config().save_pretrained(path)
    (path / "modelopt_state.pth").touch()
    monkeypatch.setitem(sys.modules, "modelopt.torch.opt", None)
    with pytest.raises(ImportError, match="contains ModelOpt quantizer state"):
        HfModelHandler(path, task="text-generation").load_model()


def test_modelopt_accepts_every_installed_native_preset(hf_model, monkeypatch, tmp_path):
    pytest.importorskip("modelopt", minversion="0.48.0")
    import modelopt.torch.export as export
    import modelopt.torch.quantization as mtq
    from modelopt.recipe.presets import QUANT_CFG_CHOICES
    from modelopt.torch.quantization import config

    quantize = Mock(return_value=hf_model.load_model.return_value)
    monkeypatch.setattr(mtq, "quantize", quantize)
    monkeypatch.setattr(config, "need_calibration", Mock(return_value=False))
    monkeypatch.setattr(export, "export_hf_checkpoint", Mock())
    monkeypatch.setattr("olive.passes.pytorch.modelopt.inherit_hf_from_hf", Mock(return_value=hf_model))
    assert QUANT_CFG_CHOICES
    for name, native_config in QUANT_CFG_CHOICES.items():
        _create_pass(quantization_config=name).run(hf_model, str(tmp_path / name))
        resolved = quantize.call_args.args[1]
        assert resolved == native_config
        assert resolved is not native_config
    assert quantize.call_count == len(QUANT_CFG_CHOICES)


@pytest.mark.parametrize(
    "preset", ["fp8", "int8_smoothquant", "int4_awq", "nvfp4", "mxfp4", "mxfp6", "mxfp8", "mxint8"]
)
def test_modelopt_native_precision_roundtrip_on_cpu(preset, monkeypatch, tmp_path):
    pytest.importorskip("modelopt", minversion="0.48.0")
    from modelopt.torch.quantization.nn import TensorQuantizer
    from transformers import AutoModelForCausalLM, LlamaConfig

    config = LlamaConfig.from_dict(
        {
            "vocab_size": 256,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
        }
    )
    source = tmp_path / "input"
    AutoModelForCausalLM.from_config(config).save_pretrained(source)
    model = HfModelHandler(source, task="text-generation")
    model.save_metadata = Mock()
    loaded_models = []
    load_model = model.load_model

    def capture_model(**kwargs):
        torch_model = load_model(**kwargs)
        loaded_models.append(torch_model)
        return torch_model

    model.load_model = capture_model
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.int64)}
    monkeypatch.setattr("olive.passes.pytorch.modelopt.get_calibration_dataset", Mock(return_value=[batch]))
    result = _create_pass(quantization_config=preset, export_format="modelopt").run(model, str(tmp_path / "output"))
    assert (Path(result.model_path) / "modelopt_state.pth").is_file()
    restored = result.load_model(cache_model=False).eval()
    quantized = loaded_models[0]
    original_quantizers = {
        name: (module.num_bits, module.is_enabled, module.block_sizes)
        for name, module in quantized.named_modules()
        if isinstance(module, TensorQuantizer)
    }
    restored_quantizers = {
        name: (module.num_bits, module.is_enabled, module.block_sizes)
        for name, module in restored.named_modules()
        if isinstance(module, TensorQuantizer)
    }
    assert original_quantizers == restored_quantizers
    assert any(enabled for _, enabled, _ in restored_quantizers.values())
    original_state = quantized.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for name, tensor in original_state.items():
        torch.testing.assert_close(tensor, restored_state[name])
    if preset in ("fp8", "int8_smoothquant", "int4_awq"):
        with torch.no_grad():
            assert torch.isfinite(restored(**batch, use_cache=False).logits).all()


def test_modelopt_native_nvfp4_exports_packed_qwen_checkpoint_on_cpu(monkeypatch, tmp_path):
    pytest.importorskip("modelopt", minversion="0.48.0")
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3_5Config

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
    source = tmp_path / "input"
    AutoModelForCausalLM.from_config(Qwen3_5Config.from_dict({"text_config": text_config})).save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    tokenizer.save_pretrained(source)
    model = HfModelHandler(source, task="text-generation")
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.int64)}
    monkeypatch.setattr("olive.passes.pytorch.modelopt.get_calibration_dataset", Mock(return_value=[batch]))
    result = _create_pass(quantization_config="nvfp4").run(model, str(tmp_path / "output"))
    output = Path(result.model_path)
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "qwen3_5_text"
    assert config["quantization_config"]["quant_algo"] == "NVFP4"
    assert config["quantization_config"]["group_size"] == 16
    assert (output / "hf_quant_config.json").is_file()
    assert not (output / "modelopt_state.pth").exists()
    assert (output / "tokenizer_config.json").is_file()
    assert (output / "tokenizer.json").is_file()
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        weights = [
            checkpoint.get_tensor(name) for name in sorted(checkpoint.keys()) if name.endswith(".mlp.up_proj.weight")
        ]
    assert len(weights) == 2
    assert all(weight.dtype == torch.uint8 and weight.numel() == 32 * 64 // 2 for weight in weights)
