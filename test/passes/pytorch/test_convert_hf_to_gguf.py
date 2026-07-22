# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from olive.passes.pytorch.convert_hf_to_gguf import ConvertHfToGGUF


def test_convert_hf_to_gguf_skips_when_missing_source(tmp_path):
    pass_instance = ConvertHfToGGUF.__new__(ConvertHfToGGUF)
    model = SimpleNamespace(test_model_path=str(tmp_path / "missing"), model_attributes=None)
    config = SimpleNamespace(
        llama_cpp_env_path=str(tmp_path / "llama_env"),
        reference_model_path=str(tmp_path / "missing"),
        gguf_file_name="model.gguf",
    )

    result = pass_instance._run_for_config(model, config, str(tmp_path / "out"))
    assert result is model


def test_convert_hf_to_gguf_uses_existing_gguf(tmp_path):
    source = tmp_path / "test_model"
    source.mkdir(parents=True, exist_ok=True)
    gguf_path = source / "model.gguf"
    gguf_path.write_text("ok")

    pass_instance = ConvertHfToGGUF.__new__(ConvertHfToGGUF)
    model = SimpleNamespace(test_model_path=str(source), model_attributes={})
    config = SimpleNamespace(
        llama_cpp_env_path=str(tmp_path / "llama_env"),
        reference_model_path=str(source),
        gguf_file_name="model.gguf",
    )

    result = pass_instance._run_for_config(model, config, str(tmp_path / "out"))
    assert result.model_attributes["reference_gguf_model_path"] == str(gguf_path)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="llama_env is not available on Windows CI")
def test_convert_hf_to_gguf_runs_conversion(tmp_path):
    source = tmp_path / "test_model"
    source.mkdir(parents=True, exist_ok=True)
    env = tmp_path / "llama_env"
    env.mkdir(parents=True, exist_ok=True)
    (env / "convert_hf_to_gguf.py").write_text("")
    (env / "conversion").mkdir(parents=True, exist_ok=True)

    pass_instance = ConvertHfToGGUF.__new__(ConvertHfToGGUF)
    python_path = Path(pass_instance._get_python_executable(env))
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("")
    model = SimpleNamespace(test_model_path=str(source), model_attributes={})
    config = SimpleNamespace(
        llama_cpp_env_path=str(env),
        reference_model_path=str(source),
        gguf_file_name="model.gguf",
    )

    with patch("olive.passes.pytorch.convert_hf_to_gguf.subprocess.run") as mock_run:
        result = pass_instance._run_for_config(model, config, str(tmp_path / "out"))

    assert mock_run.call_count == 1
    assert Path(result.model_attributes["reference_gguf_model_path"]).name == "model.gguf"
