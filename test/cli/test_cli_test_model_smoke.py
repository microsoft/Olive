# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from olive.cli.base import TEST_OUTPUT_MARKER_FILE
from olive.common.hf.utils import TEST_MODEL_MARKER_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_IDS = (
    "local/tiny-random-llama-a",
    "local/tiny-random-llama-b",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
MAX_ARTIFACT_SIZE_BYTES = 1024 * 1024

# Supported exporters for the discrepancy test
EXPORTER_MODEL_BUILDER = "model_builder"
EXPORTER_MOBIUS = "mobius"
EXPORTER_TORCH = "torch_export"

_HAS_MOBIUS = importlib.util.find_spec("mobius") is not None


def _save_local_tiny_llama(model_path: Path):
    model = LlamaForCausalLM(
        LlamaConfig.from_dict(
            {
                "vocab_size": 32,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "max_position_embeddings": 64,
            }
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


def _run_cli_main(args):
    from olive.cli.launcher import main as cli_main

    cli_main(args)


def _run_documented_test_model_smoke_flow(tmp_path: Path, model_id: str):
    model_name = model_id.replace("/", "--")
    model_path = tmp_path / "models" / model_name
    config_output_dir = tmp_path / f"{model_name}-test"
    test_model_dir = tmp_path / f"{model_name}-test-model"
    run_output_dir = tmp_path / f"{model_name}-test-run"

    _save_local_tiny_llama(model_path)
    # optimize -m arnir0/Tiny-LLM --device cpu --provider CPUExecutionProvider --precision int4 --output_path dump --dry_run
    _run_cli_main(
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
    # run --config dump/config.json --test dump/test --output_path dump/run
    _run_cli_main(
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

    return config_path, test_model_dir, run_output_dir


class TestCliTestModelSmoke(unittest.TestCase):
    model_ids = DEFAULT_MODEL_IDS
    exporters = (EXPORTER_MODEL_BUILDER,)
    workdir = None

    def test_documented_test_model_smoke_flow(self):
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_smoke_flows(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_smoke_flows(workdir)

    def _assert_smoke_flows(self, tmp_path: Path):
        expected_test_model_files = {
            "config.json",
            "generation_config.json",
            "model.safetensors",
            TEST_MODEL_MARKER_FILE,
        }
        expected_run_output_files = {
            "config.json",
            "genai_config.json",
            "generation_config.json",
            "model.onnx",
            "model_config.json",
            TEST_OUTPUT_MARKER_FILE,
            "tokenizer.json",
            "tokenizer_config.json",
        }
        for model_id in self.model_ids:
            with self.subTest(model_id=model_id):
                config_path, test_model_dir, run_output_dir = _run_documented_test_model_smoke_flow(tmp_path, model_id)
                assert config_path.exists()
                assert self._list_relative_files(test_model_dir) == expected_test_model_files
                run_output_files = self._list_relative_files(run_output_dir)
                assert expected_run_output_files.issubset(run_output_files)
                self._assert_file_size_below_limit(test_model_dir / "model.safetensors")
                if "model.onnx.data" in run_output_files:
                    self._assert_file_size_below_limit(run_output_dir / "model.onnx.data")

    def test_model_discrepancy(self):
        """Verify that OnnxDiscrepancyCheck runs successfully with the configured exporter."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_discrepancy(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_discrepancy(workdir)

    def _assert_discrepancy(self, tmp_path: Path):
        for exporter in self.exporters:
            if exporter == EXPORTER_MOBIUS and not _HAS_MOBIUS:
                self.fail(
                    "Requested exporter 'mobius' but mobius-ai is not installed. Install mobius-ai or remove '--exporter mobius'."
                )
            for model_id in self.model_ids:
                with self.subTest(model_id=model_id, exporter=exporter):
                    if exporter == EXPORTER_MODEL_BUILDER:
                        self._assert_discrepancy_model_builder(tmp_path, model_id)
                    elif exporter == EXPORTER_MOBIUS:
                        self._assert_discrepancy_mobius(tmp_path, model_id)
                    elif exporter == EXPORTER_TORCH:
                        self._assert_discrepancy_torch_export(tmp_path, model_id)
                    else:
                        self.fail(f"Unknown exporter: {exporter!r}")

    @staticmethod
    def _run_discrepancy_with_test(config_path: Path, test_model_dir: Path, run_output_dir: Path):
        _run_cli_main(
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

    def _assert_discrepancy_model_builder(self, tmp_path: Path, model_id: str):
        model_name = model_id.replace("/", "--")
        model_path = tmp_path / "models" / f"{model_name}-disc"
        config_output_dir = tmp_path / f"{model_name}-disc-cfg"
        test_model_dir = tmp_path / f"{model_name}-disc-test-model"
        run_output_dir = tmp_path / f"{model_name}-disc-run"

        _save_local_tiny_llama(model_path)
        _run_cli_main(
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

        # Run with --test; OnnxDiscrepancyCheck is auto-injected and reports discrepancy metrics
        self._run_discrepancy_with_test(config_path, test_model_dir, run_output_dir)

    def _assert_discrepancy_mobius(self, tmp_path: Path, model_id: str):
        model_name = model_id.replace("/", "--")
        model_path = tmp_path / "models" / f"{model_name}-mobius-disc"
        test_model_dir = tmp_path / f"{model_name}-mobius-disc-test-model"
        run_output_dir = tmp_path / f"{model_name}-mobius-disc-run"

        _save_local_tiny_llama(model_path)

        run_config = {
            "input_model": {
                "type": "HfModel",
                "model_path": str(model_path),
            },
            "systems": {
                "local_system": {
                    "type": "LocalSystem",
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
                }
            },
            "output_dir": str(run_output_dir),
            "host": "local_system",
            "target": "local_system",
            "passes": {
                "mobius_builder": {
                    "type": "MobiusBuilder",
                    "precision": "fp32",
                    "runtime": "none",
                },
                "discrepancy_check": {
                    "type": "OnnxDiscrepancyCheck",
                    "reference_model_path": str(test_model_dir),
                },
            },
        }
        config_path = tmp_path / f"{model_name}-mobius-disc-config.json"
        config_path.write_text(json.dumps(run_config, indent=2))

        self._run_discrepancy_with_test(config_path, test_model_dir, run_output_dir)

    def _assert_discrepancy_torch_export(self, tmp_path: Path, model_id: str):
        model_name = model_id.replace("/", "--")
        model_path = tmp_path / "models" / f"{model_name}-torch-disc"
        test_model_dir = tmp_path / f"{model_name}-torch-disc-test-model"
        run_output_dir = tmp_path / f"{model_name}-torch-disc-run"

        _save_local_tiny_llama(model_path)

        run_config = {
            "input_model": {
                "type": "HfModel",
                "model_path": str(model_path),
            },
            "systems": {
                "local_system": {
                    "type": "LocalSystem",
                    "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
                }
            },
            "output_dir": str(run_output_dir),
            "host": "local_system",
            "target": "local_system",
            "passes": {
                "conversion": {
                    "type": "OnnxConversion",
                    "use_dynamo_exporter": False,
                }
            },
        }
        config_path = tmp_path / f"{model_name}-torch-disc-config.json"
        config_path.write_text(json.dumps(run_config, indent=2))
        self._run_discrepancy_with_test(config_path, test_model_dir, run_output_dir)

    def _assert_file_size_below_limit(self, path: Path):
        assert path.exists()
        assert path.stat().st_size < MAX_ARTIFACT_SIZE_BYTES

    @staticmethod
    def _list_relative_files(path: Path):
        return {file_path.relative_to(path).as_posix() for file_path in path.rglob("*") if file_path.is_file()}


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workdir")
    parser.add_argument("--model-id", dest="model_ids", action="append")
    parser.add_argument(
        "--exporter",
        dest="exporters",
        action="append",
        choices=[EXPORTER_MODEL_BUILDER, EXPORTER_MOBIUS, EXPORTER_TORCH],
        help="Exporter(s) to test for discrepancy check (can be repeated).",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    parsed_args, remaining = _parse_args()
    if parsed_args.workdir:
        TestCliTestModelSmoke.workdir = Path(parsed_args.workdir)
    if parsed_args.model_ids:
        TestCliTestModelSmoke.model_ids = tuple(parsed_args.model_ids)
    if parsed_args.exporters:
        TestCliTestModelSmoke.exporters = tuple(parsed_args.exporters)
    unittest.main(argv=[__file__, *remaining])
