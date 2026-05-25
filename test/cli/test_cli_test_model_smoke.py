# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from olive.cli.launcher import main as cli_main


@patch("huggingface_hub.repo_exists", return_value=True)
def test_documented_test_model_smoke_flow(mock_repo_exists, tmp_path):
    from olive.model.config.model_config import ModelConfig
    from olive.model.handler.hf import HfModelHandler
    from olive.passes.onnx.model_builder import ModelBuilder
    from olive.passes.pytorch.gptq import Gptq

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    config_output_dir = tmp_path / "qwen-smoke"
    test_model_dir = tmp_path / "qwen-test-model"
    run_output_dir = tmp_path / "qwen-smoke-run"

    cli_main(
        [
            "optimize",
            "-m",
            model_id,
            "--device",
            "cpu",
            "--provider",
            "CPUExecutionProvider",
            "--precision",
            "int4",
            "--output_path",
            str(config_output_dir),
            "--dry_run",
        ]
    )

    config_path = config_output_dir / "config.json"
    assert config_path.exists()
    assert mock_repo_exists.called

    def fake_load_model(handler, *args, **kwargs):
        Path(handler.test_model_path).mkdir(parents=True, exist_ok=True)
        (Path(handler.test_model_path) / "config.json").write_text(json.dumps({"model_type": "llama"}))
        return MagicMock()

    def fake_gptq_run(self, model, pass_config, output_model_path):
        del pass_config, output_model_path
        return model

    def fake_create_model(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / kwargs["filename"]).write_text("dummy ONNX file")
        (output_dir / "genai_config.json").write_text("{}")

    fake_builder = types.ModuleType("onnxruntime_genai.models.builder")
    fake_builder.create_model = MagicMock(side_effect=fake_create_model)
    fake_models = types.ModuleType("onnxruntime_genai.models")
    fake_models.builder = fake_builder
    fake_ort_genai = types.ModuleType("onnxruntime_genai")
    fake_ort_genai.models = fake_models
    fake_ort_genai.__version__ = "0.10.0"
    mock_cfg = MagicMock()
    mock_cfg.to_dict.return_value = {}

    with (
        patch.object(ModelConfig, "get_model_identifier", return_value="tiny-random-llama"),
        patch.object(HfModelHandler, "get_hf_model_config", return_value=mock_cfg),
        patch.object(HfModelHandler, "load_model", new=fake_load_model),
        patch.object(HfModelHandler, "save_metadata", return_value=[]),
        patch.object(Gptq, "_run_for_config", autospec=True, side_effect=fake_gptq_run),
        patch.object(ModelBuilder, "maybe_patch_quant", return_value=None),
        patch.dict(
            sys.modules,
            {
                "onnxruntime_genai": fake_ort_genai,
                "onnxruntime_genai.models": fake_models,
                "onnxruntime_genai.models.builder": fake_builder,
            },
        ),
    ):
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--test",
                str(test_model_dir),
                "--output_path",
                str(run_output_dir),
            ]
        )

    assert (test_model_dir / "config.json").exists()
    assert (run_output_dir / "model.onnx").exists()
    assert (run_output_dir / "genai_config.json").exists()
