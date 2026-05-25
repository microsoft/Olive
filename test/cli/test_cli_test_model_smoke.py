# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from olive.cli.launcher import main as cli_main


def _save_local_tiny_llama(model_path: Path):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(model_path)

    tokenizer = Tokenizer(
        WordLevel(
            vocab={"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4},
            unk_token="<pad>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    ).save_pretrained(model_path)


def _set_offline_gptq_data_config(config_path: Path):
    config = json.loads(config_path.read_text())
    config["passes"]["gptq"]["data_config"] = {
        "name": "test_gptq_dummy_data",
        "type": "DummyDataContainer",
        "load_dataset_config": {
            "type": "dummy_dataset",
            "params": {
                "input_names": ["input_ids", "attention_mask"],
                "input_shapes": [[1, 8], [1, 8]],
                "input_types": ["int64", "int64"],
                "max_samples": 1,
            },
        },
        "pre_process_data_config": {"type": "skip_pre_process"},
        "post_process_data_config": {"type": "skip_post_process"},
    }
    config_path.write_text(json.dumps(config, indent=2))


def test_documented_test_model_smoke_flow(tmp_path):
    model_path = tmp_path / "tiny-random-llama"
    config_output_dir = tmp_path / "qwen-smoke"
    test_model_dir = tmp_path / "qwen-test-model"
    run_output_dir = tmp_path / "qwen-smoke-run"

    _save_local_tiny_llama(model_path)
    cli_main(
        [
            "optimize",
            "-m",
            str(model_path),
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
    _set_offline_gptq_data_config(config_path)
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
    assert list(Path(run_output_dir).rglob("*.onnx"))
    assert (run_output_dir / "genai_config.json").exists()
