# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

DEFAULT_WHISPER_MODEL_IDS = (
    "openai/whisper-tiny",
    "microsoft/whisper-base",
)
MAX_ARTIFACT_SIZE_BYTES = 1024 * 1024


def _run_cli_main(args):
    from olive.cli.launcher import main as cli_main

    cli_main(args)


def _save_local_tiny_whisper(model_path: Path):
    from transformers import PreTrainedTokenizerFast, WhisperConfig, WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration(
        WhisperConfig.from_dict(
            {
                "num_mel_bins": 80,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "decoder_layers": 2,
                "decoder_attention_heads": 4,
                "d_model": 64,
                "encoder_ffn_dim": 128,
                "decoder_ffn_dim": 128,
                "max_source_positions": 16,
                "max_target_positions": 16,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "decoder_start_token_id": 1,
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


def _run_whisper_capture_onnx_flow(tmp_path: Path, model_id: str, precision: str = "fp32"):
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
            precision,
            "--output_path",
            str(run_output_dir),
        ]
    )

    return run_output_dir


class TestCliWhisperSmoke(unittest.TestCase):
    """Smoke tests for Whisper encoder-decoder models exported via capture-onnx-graph."""

    model_ids = DEFAULT_WHISPER_MODEL_IDS
    precision = "fp32"
    workdir = None

    def test_whisper_capture_onnx_graph(self):
        """Verify that Whisper encoder and decoder are exported to ONNX successfully."""
        if self.workdir is None:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
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
                run_output_dir = _run_whisper_capture_onnx_flow(tmp_path, model_id, self.precision)
                output_files = self._list_relative_files(run_output_dir)
                assert expected_encoder_file in output_files, (
                    f"Expected {expected_encoder_file} in output, got: {output_files}"
                )
                assert expected_decoder_file in output_files, (
                    f"Expected {expected_decoder_file} in output, got: {output_files}"
                )
                self._assert_file_size_below_limit(run_output_dir / expected_encoder_file)
                self._assert_file_size_below_limit(run_output_dir / expected_decoder_file)

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
    return parser.parse_known_args()


if __name__ == "__main__":
    parsed_args, remaining = _parse_args()
    if parsed_args.workdir:
        TestCliWhisperSmoke.workdir = Path(parsed_args.workdir)
    if parsed_args.model_ids:
        TestCliWhisperSmoke.model_ids = tuple(parsed_args.model_ids)
    unittest.main(argv=[__file__, *remaining])
