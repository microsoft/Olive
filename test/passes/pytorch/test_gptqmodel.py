# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import gptqmodel as gptqmodel_module
from olive.passes.pytorch.gptqmodel import GptqModel, _get_embed_quant_config


class _QuantizeEmbed:
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"

    def __new__(cls, value):
        if value not in {cls.INPUT, cls.OUTPUT, cls.BOTH}:
            raise ValueError(value)
        return value


class _QuantizeEmbedConfig:
    def __init__(self, embed_quant_mode, embed_only):
        self.embed_quant_mode = embed_quant_mode
        self.embed_only = embed_only


class _QuantizeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.lm_head = kwargs["lm_head"]


class _Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class _BaseGPTQModel:
    instance = None

    def __init__(self, model, _quantized, quantize_config, **kwargs):
        self.model = model
        self.quantize_config = quantize_config
        self.init_kwargs = kwargs
        self.quantize_kwargs = None
        self.runtime_lm_head = None
        _BaseGPTQModel.instance = self

    def quantize(self, dataset, **kwargs):
        self.quantize_kwargs = {"dataset": dataset, **kwargs}
        self.runtime_lm_head = self.quantize_config.lm_head

    def save_quantized(self, output_model_path):
        output_path = Path(output_model_path)
        output_path.mkdir(parents=True)
        config = {"quantization_config": {"lm_head": self.quantize_config.lm_head}}
        (output_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


class _RegisteredGPTQModel(_BaseGPTQModel):
    pass


def _install_fake_gptqmodel(monkeypatch, base_model_name="BaseGPTQModel"):
    _BaseGPTQModel.instance = None
    gptqmodel = ModuleType("gptqmodel")
    gptqmodel.QuantizeConfig = _QuantizeConfig
    gptqmodel.QuantizeEmbed = _QuantizeEmbed
    gptqmodel.QuantizeEmbedConfig = _QuantizeEmbedConfig
    models = ModuleType("gptqmodel.models")
    auto = ModuleType("gptqmodel.models.auto")
    setattr(auto, base_model_name, _BaseGPTQModel)
    auto.MODEL_MAP = {"test": _RegisteredGPTQModel}
    models.auto = auto
    constants = ModuleType("gptqmodel.models._const")
    constants.normalize_device = _Device
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)
    monkeypatch.setitem(sys.modules, "gptqmodel.models", models)
    monkeypatch.setitem(sys.modules, "gptqmodel.models.auto", auto)
    monkeypatch.setitem(sys.modules, "gptqmodel.models._const", constants)


def test_gptqmodel_embedding_quantization_is_disabled_by_default(monkeypatch):
    _install_fake_gptqmodel(monkeypatch)

    quantizer = create_pass_from_dict(
        GptqModel,
        {},
        disable_search=True,
    )
    assert _get_embed_quant_config(quantizer.config) is None


@pytest.mark.parametrize("base_model_name", ["BaseGPTQModel", "BaseQModel"])
@pytest.mark.parametrize("model_type", ["test", "unknown"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("embed_quant_mode", [None, "input", "output", "both"])
def test_gptqmodel_forwards_quantization_config(
    monkeypatch, tmp_path, base_model_name, model_type, device, embed_quant_mode
):
    _install_fake_gptqmodel(monkeypatch, base_model_name)
    dataset = object()
    tokenizer = object()
    output_model = MagicMock()
    output_model.model_attributes = {}
    monkeypatch.setattr(gptqmodel_module, "get_calibration_dataset", lambda *_: dataset)
    monkeypatch.setattr(gptqmodel_module, "get_tokenizer", lambda *_: tokenizer)
    monkeypatch.setattr(gptqmodel_module, "inherit_hf_from_hf", lambda *_args, **_kwargs: output_model)

    model = MagicMock()
    model.model_path = "input-model"
    model.load_kwargs = None
    model.model_attributes = {}
    pytorch_model = MagicMock()
    pytorch_model.config.model_type = model_type
    model.load_model.return_value = pytorch_model

    quantizer = create_pass_from_dict(
        GptqModel, {"device": device, "embed_quant_mode": embed_quant_mode}, disable_search=True
    )
    result = quantizer.run(model, str(tmp_path / "output"))

    assert result is output_model
    expected_class = _RegisteredGPTQModel if model_type == "test" else _BaseGPTQModel
    assert isinstance(_BaseGPTQModel.instance, expected_class)
    assert isinstance(_BaseGPTQModel.instance, _RegisteredGPTQModel) is (model_type == "test")
    assert _BaseGPTQModel.instance.quantize_config.kwargs["device"] is _Device(device)
    assert _BaseGPTQModel.instance.runtime_lm_head is False
    assert _BaseGPTQModel.instance.quantize_kwargs["dataset"] is dataset
    assert _BaseGPTQModel.instance.quantize_kwargs["tokenizer"] is tokenizer
    embed_config = _BaseGPTQModel.instance.quantize_kwargs["embed_quant_config"]
    if embed_quant_mode is None:
        assert embed_config is None
    else:
        assert embed_config.embed_quant_mode == embed_quant_mode
        assert embed_config.embed_only is False
    saved_config = json.loads((tmp_path / "output" / "config.json").read_text(encoding="utf-8"))
    assert saved_config["quantization_config"]["lm_head"] is (embed_quant_mode in {"output", "both"})


def test_gptqmodel_embedding_quantization_can_be_enabled(monkeypatch):
    _install_fake_gptqmodel(monkeypatch)

    quantizer = create_pass_from_dict(
        GptqModel,
        {"embed_quant_mode": "both", "embed_only": False},
        disable_search=True,
    )
    embed_config = _get_embed_quant_config(quantizer.config)

    assert embed_config.embed_quant_mode == _QuantizeEmbed.BOTH
    assert embed_config.embed_only is False
