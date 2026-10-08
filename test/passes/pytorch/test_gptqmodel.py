# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

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


class _BaseGPTQModel:
    instance = None

    def __init__(self, model, _quantized, quantize_config, **kwargs):
        self.model = model
        self.quantize_config = quantize_config
        self.init_kwargs = kwargs
        self.quantize_kwargs = None
        _BaseGPTQModel.instance = self

    def quantize(self, dataset, **kwargs):
        self.quantize_kwargs = {"dataset": dataset, **kwargs}

    @staticmethod
    def save_quantized(output_model_path):
        output_path = Path(output_model_path)
        output_path.mkdir(parents=True)
        (output_path / "config.json").write_text(json.dumps({"quantization_config": {}}), encoding="utf-8")


def _install_fake_gptqmodel(monkeypatch):
    _BaseGPTQModel.instance = None
    gptqmodel = ModuleType("gptqmodel")
    gptqmodel.QuantizeConfig = _QuantizeConfig
    gptqmodel.QuantizeEmbed = _QuantizeEmbed
    gptqmodel.QuantizeEmbedConfig = _QuantizeEmbedConfig
    models = ModuleType("gptqmodel.models")
    auto = ModuleType("gptqmodel.models.auto")
    auto.BaseGPTQModel = _BaseGPTQModel
    auto.MODEL_MAP = {"test": _BaseGPTQModel}
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)
    monkeypatch.setitem(sys.modules, "gptqmodel.models", models)
    monkeypatch.setitem(sys.modules, "gptqmodel.models.auto", auto)


def test_gptqmodel_embedding_quantization_is_disabled_by_default(monkeypatch):
    _install_fake_gptqmodel(monkeypatch)

    quantizer = create_pass_from_dict(
        GptqModel,
        {},
        disable_search=True,
    )
    assert _get_embed_quant_config(quantizer.config) is None


def test_gptqmodel_forwards_disabled_embedding_config(monkeypatch, tmp_path):
    _install_fake_gptqmodel(monkeypatch)
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
    pytorch_model.config.model_type = "test"
    model.load_model.return_value = pytorch_model

    quantizer = create_pass_from_dict(GptqModel, {}, disable_search=True)
    result = quantizer.run(model, str(tmp_path / "output"))

    assert result is output_model
    assert _BaseGPTQModel.instance.quantize_kwargs["dataset"] is dataset
    assert _BaseGPTQModel.instance.quantize_kwargs["tokenizer"] is tokenizer
    assert _BaseGPTQModel.instance.quantize_kwargs["embed_quant_config"] is None


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
