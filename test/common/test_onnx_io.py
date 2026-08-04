# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

from olive.common.onnx_io import get_genai_decoder_config

DECODER_CONFIG = {"head_size": 64, "num_key_value_heads": 2}


def _save_genai_config(model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "genai_config.json").write_text(json.dumps({"model": {"decoder": DECODER_CONFIG}}))


def test_get_genai_decoder_config_returns_decoder_section_for_model_file(tmp_path):
    _save_genai_config(tmp_path)

    assert get_genai_decoder_config(tmp_path / "model.onnx") == DECODER_CONFIG


def test_get_genai_decoder_config_returns_decoder_section_for_model_dir(tmp_path):
    _save_genai_config(tmp_path)

    assert get_genai_decoder_config(tmp_path) == DECODER_CONFIG


def test_get_genai_decoder_config_returns_none_when_config_is_missing(tmp_path):
    assert get_genai_decoder_config(tmp_path / "model.onnx") is None
