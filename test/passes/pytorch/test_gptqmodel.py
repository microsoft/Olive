# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from types import ModuleType
from unittest.mock import MagicMock

from olive.passes.olive_pass import create_pass_from_dict
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


def test_gptqmodel_embedding_quantization_defaults_to_full_model(monkeypatch):
    gptqmodel = ModuleType("gptqmodel")
    gptqmodel.QuantizeEmbed = _QuantizeEmbed
    gptqmodel.QuantizeEmbedConfig = _QuantizeEmbedConfig
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)

    quantizer = create_pass_from_dict(
        GptqModel,
        {},
        disable_search=True,
    )
    embed_config = _get_embed_quant_config(quantizer.config)

    assert embed_config.embed_quant_mode == _QuantizeEmbed.BOTH
    assert embed_config.embed_only is False


def test_gptqmodel_without_embedding_quantization():
    config = MagicMock(embed_quant_mode=None)
    assert _get_embed_quant_config(config) is None
