# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from olive.cli.base import TEST_OUTPUT_MARKER_FILE
from olive.common.hf.utils import TEST_MODEL_MARKER_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_IDS = (
    "local/tiny-random-llama-a",
    "local/tiny-random-llama-b",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
)
DEFAULT_WHISPER_MODEL_IDS = (
    "openai/whisper-tiny",
    "microsoft/whisper-base",
)
MAX_ARTIFACT_SIZE_BYTES = 1024 * 1024


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


def _save_local_tiny_whisper(model_path: Path):
    model = WhisperForConditionalGeneration(
        WhisperConfig(
            vocab_size=32,
            num_mel_bins=80,
            encoder_layers=2,
            encoder_attention_heads=4,
            decoder_layers=2,
            decoder_attention_heads=4,
            d_model=64,
            encoder_ffn_dim=128,
            decoder_ffn_dim=128,
            max_source_positions=16,
            max_target_positions=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
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

    def test_genai_inference(self):
        """Verify that the optimized ONNX model can be loaded and run with onnxruntime-genai."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_genai_inference(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_genai_inference(workdir)

    def _assert_genai_inference(self, tmp_path: Path):
        import onnxruntime_genai as og

        input_ids = [1, 3, 4]  # [bos, hello, world]
        for model_id in self.model_ids:
            with self.subTest(model_id=model_id):
                _, _, run_output_dir = _run_documented_test_model_smoke_flow(tmp_path, model_id)
                model_path = tmp_path / "models" / model_id.replace("/", "--")
                transformers_token = _get_transformers_first_token(model_path, input_ids)
                # Load the quantized model with genai and generate one token
                config = og.Config(str(run_output_dir))
                config.clear_providers()
                model = og.Model(config)
                params = og.GeneratorParams(model)
                params.set_search_options(
                    max_length=len(input_ids) + 1, do_sample=False, past_present_share_buffer=False
                )
                generator = og.Generator(model, params)
                generator.append_tokens([input_ids])
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                assert transformers_token == token, (
                    f"First token mismatch for {model_id}: transformers={transformers_token}, genai={token}"
                )

    def test_model_discrepancy(self):
        """Verify that OnnxDiscrepancyCheck runs successfully when auto-injected via --test."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_discrepancy(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_discrepancy(workdir)

    def _assert_discrepancy(self, tmp_path: Path):
        for model_id in self.model_ids:
            with self.subTest(model_id=model_id):
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

                # Run with --test; OnnxDiscrepancyCheck is auto-injected and reports discrepancy metrics (fails only if thresholds are configured)
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

    def _assert_file_size_below_limit(self, path: Path):
        assert path.exists()
        assert path.stat().st_size < MAX_ARTIFACT_SIZE_BYTES

    @staticmethod
    def _list_relative_files(path: Path):
        return {file_path.relative_to(path).as_posix() for file_path in path.rglob("*") if file_path.is_file()}


def _run_whisper_capture_onnx_flow(tmp_path: Path, model_id: str):
    """Export a tiny local Whisper model to ONNX via capture-onnx-graph with Model Builder."""
    model_name = model_id.replace("/", "--")
    model_path = tmp_path / "models" / model_name
    run_output_dir = tmp_path / f"{model_name}-onnx"

    _save_local_tiny_whisper(model_path)
    _run_cli_main(
        [
            "capture-onnx-graph",
            "-m",
            str(model_path),
            "--use_model_builder",
            "--precision",
            "fp32",
            "--output_path",
            str(run_output_dir),
        ]
    )

    return run_output_dir


class TestCliWhisperSmoke(unittest.TestCase):
    """Smoke tests for Whisper encoder-decoder models exported via capture-onnx-graph."""

    model_ids = DEFAULT_WHISPER_MODEL_IDS
    workdir = None

    def test_whisper_capture_onnx_graph(self):
        """Verify that Whisper encoder and decoder are exported to ONNX successfully."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_whisper_export(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_whisper_export(workdir)

    def _assert_whisper_export(self, tmp_path: Path):
        expected_encoder_file = "encoder.onnx"
        expected_decoder_file = "decoder.onnx"
        for model_id in self.model_ids:
            with self.subTest(model_id=model_id):
                run_output_dir = _run_whisper_capture_onnx_flow(tmp_path, model_id)
                output_files = self._list_relative_files(run_output_dir)
                assert expected_encoder_file in output_files, (
                    f"Expected {expected_encoder_file} in output, got: {output_files}"
                )
                assert expected_decoder_file in output_files, (
                    f"Expected {expected_decoder_file} in output, got: {output_files}"
                )
                self._assert_file_size_below_limit(run_output_dir / expected_encoder_file)
                self._assert_file_size_below_limit(run_output_dir / expected_decoder_file)

    def test_whisper_genai_inference(self):
        """Verify that the exported Whisper ONNX model can be loaded and run with onnxruntime-genai."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_whisper_genai_inference(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_whisper_genai_inference(workdir)

    def _assert_whisper_genai_inference(self, tmp_path: Path):
        import io

        import numpy as np
        import onnxruntime_genai as og
        import soundfile as sf

        for model_id in self.whisper_model_ids:
            with self.subTest(model_id=model_id):
                run_output_dir = _run_whisper_capture_onnx_flow(tmp_path, model_id)

                # Load model with genai multimodal processor
                config = og.Config(str(run_output_dir))
                config.clear_providers()
                og_model = og.Model(config)
                processor = og_model.create_multimodal_processor()

                # Create 1 second of silence as WAV bytes
                sample_rate = 16000
                audio_samples = np.zeros(sample_rate, dtype=np.float32)
                buffer = io.BytesIO()
                sf.write(buffer, audio_samples, samplerate=sample_rate, format="WAV")
                audios = og.Audios.open_bytes(buffer.getvalue())

                # Generate one decoder token
                prompt = "<|startoftranscript|>"
                inputs = processor([prompt], audios=audios)
                params = og.GeneratorParams(og_model)
                params.set_search_options(max_length=4, do_sample=False, batch_size=1)
                generator = og.Generator(og_model, params)
                generator.set_inputs(inputs)
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                assert isinstance(token, int), f"Expected int token, got {type(token)}"

    def _assert_file_size_below_limit(self, path: Path):
        assert path.exists()
        assert path.stat().st_size < MAX_ARTIFACT_SIZE_BYTES

    @staticmethod
    def _list_relative_files(path: Path):
        return {file_path.relative_to(path).as_posix() for file_path in path.rglob("*") if file_path.is_file()}


def _get_transformers_first_token(model_path: Path, input_ids: list[int]) -> int:
    """Generate one token with transformers and return its id."""
    import torch

    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=1, do_sample=False)
    return output[0, len(input_ids)].item()


def _get_genai_first_token(onnx_output_dir: Path, input_ids: list[int]) -> int:
    """Generate one token with onnxruntime-genai and return its id."""
    import onnxruntime_genai as og

    config = og.Config(str(onnx_output_dir))
    config.clear_providers()
    model = og.Model(config)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(input_ids) + 1, do_sample=False, past_present_share_buffer=False)
    generator = og.Generator(model, params)
    generator.append_tokens([input_ids])
    generator.generate_next_token()
    return generator.get_next_tokens()[0]


def _export_tiny_llama_fp32(tmp_path: Path, model_id: str) -> tuple[Path, Path]:
    """Create a tiny LLaMA, export with model_builder fp32, return (model_path, onnx_output_dir)."""
    model_name = model_id.replace("/", "--")
    model_path = tmp_path / "models" / model_name
    onnx_output_dir = tmp_path / f"{model_name}-fp32-onnx"

    _save_local_tiny_llama(model_path)
    _run_cli_main(
        [
            "capture-onnx-graph",
            "-m",
            str(model_path),
            "--use_model_builder",
            "--precision",
            "fp32",
            "--output_path",
            str(onnx_output_dir),
        ]
    )

    return model_path, onnx_output_dir


def _export_tiny_whisper_fp32(tmp_path: Path, model_id: str) -> tuple[Path, Path]:
    """Create a tiny Whisper, export with model_builder fp32, return (model_path, onnx_output_dir)."""
    model_name = model_id.replace("/", "--")
    model_path = tmp_path / "models" / model_name
    onnx_output_dir = tmp_path / f"{model_name}-fp32-onnx"

    _save_local_tiny_whisper(model_path)
    _run_cli_main(
        [
            "capture-onnx-graph",
            "-m",
            str(model_path),
            "--use_model_builder",
            "--precision",
            "fp32",
            "--output_path",
            str(onnx_output_dir),
        ]
    )

    return model_path, onnx_output_dir


class TestFirstTokenDiscrepancy(unittest.TestCase):
    """Compare the first generated token between transformers and onnxruntime-genai."""

    model_ids = DEFAULT_MODEL_IDS
    whisper_model_ids = DEFAULT_WHISPER_MODEL_IDS
    workdir = None

    def test_first_token_llama(self):
        """Verify first token from transformers matches onnxruntime-genai for LLaMA models."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_first_token_llama(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_first_token_llama(workdir)

    def _assert_first_token_llama(self, tmp_path: Path):
        # Use a simple prompt: [bos, hello, world] -> token ids [1, 3, 4]
        input_ids = [1, 3, 4]
        for model_id in self.model_ids:
            with self.subTest(model_id=model_id):
                model_path, onnx_output_dir = _export_tiny_llama_fp32(tmp_path, model_id)
                transformers_token = _get_transformers_first_token(model_path, input_ids)
                genai_token = _get_genai_first_token(onnx_output_dir, input_ids)
                assert transformers_token == genai_token, (
                    f"First token mismatch for {model_id}: "
                    f"transformers={transformers_token}, genai={genai_token}"
                )

    def test_first_token_whisper(self):
        """Verify first decoder token from transformers matches onnxruntime-genai for Whisper."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._assert_first_token_whisper(Path(temp_dir))
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            self._assert_first_token_whisper(workdir)

    def _assert_first_token_whisper(self, tmp_path: Path):
        import numpy as np

        for model_id in self.whisper_model_ids:
            with self.subTest(model_id=model_id):
                model_path, onnx_output_dir = _export_tiny_whisper_fp32(tmp_path, model_id)

                # Generate first decoder token with transformers
                import torch

                whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
                whisper_model.eval()
                # Create dummy mel input: [batch, num_mel_bins, max_source_positions * 2]
                mel_length = whisper_model.config.max_source_positions * 2
                input_features = torch.randn(1, whisper_model.config.num_mel_bins, mel_length)
                with torch.no_grad():
                    output = whisper_model.generate(input_features, max_new_tokens=1, do_sample=False)
                transformers_token = output[0, 1].item()  # skip decoder_start_token

                # Generate first decoder token with onnxruntime-genai
                import onnxruntime_genai as og

                config = og.Config(str(onnx_output_dir))
                config.clear_providers()
                og_model = og.Model(config)
                processor = og_model.create_multimodal_processor()

                # Create audio input as WAV bytes
                import io
                import soundfile as sf

                # Generate silence audio matching the expected mel length
                sample_rate = 16000
                audio_samples = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
                buffer = io.BytesIO()
                sf.write(buffer, audio_samples, samplerate=sample_rate, format="WAV")
                audios = og.Audios.open_bytes(buffer.getvalue())

                prompt = "<|startoftranscript|>"
                inputs = processor([prompt], audios=audios)
                params = og.GeneratorParams(og_model)
                params.set_search_options(max_length=4, do_sample=False, batch_size=1)
                generator = og.Generator(og_model, params)
                generator.set_inputs(inputs)
                generator.generate_next_token()
                genai_token = generator.get_next_tokens()[0]

                assert transformers_token == genai_token, (
                    f"First decoder token mismatch for {model_id}: "
                    f"transformers={transformers_token}, genai={genai_token}"
                )


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workdir")
    parser.add_argument("--model-id", dest="model_ids", action="append")
    return parser.parse_known_args()


if __name__ == "__main__":
    parsed_args, remaining = _parse_args()
    if parsed_args.workdir:
        TestCliTestModelSmoke.workdir = Path(parsed_args.workdir)
        TestCliWhisperSmoke.workdir = Path(parsed_args.workdir)
        TestFirstTokenDiscrepancy.workdir = Path(parsed_args.workdir)
    if parsed_args.model_ids:
        TestCliTestModelSmoke.model_ids = tuple(parsed_args.model_ids)
    unittest.main(argv=[__file__, *remaining])
