# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from olive.data.component.pre_process_data import (
    huggingface_pre_process,
    speech_transcription_pre_process,
    tokenizer_pre_process,
    vision_vqa_pre_process,
)


class TestPreProcessData:
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock HuggingFace dataset for testing."""
        data = {
            "sentence1": ["This is a test.", "Another test sentence.", "One more example."],
            "sentence2": ["This is also a test.", "Testing again.", "Final example."],
            "label": [0, 1, 0],
        }
        return Dataset.from_dict(data)

    @patch("olive.data.component.pre_process_data.get_tokenizer")
    def test_tokenizer_pre_process(self, mock_get_tokenizer, mock_dataset):
        """Test tokenizer_pre_process function with no label processing."""
        # setup
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[101, 2023, 102], [101, 2178, 102]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        mock_get_tokenizer.return_value = mock_tokenizer

        # execute
        result = tokenizer_pre_process(
            dataset=mock_dataset,
            model_name="bert-base-uncased",
            input_cols=["sentence1", "sentence2"],
            max_samples=2,
            trust_remote_code=None,
        )

        # verify
        mock_get_tokenizer.assert_called_with("bert-base-uncased", trust_remote_code=None)
        assert result.effective_len == 2

    @patch("olive.data.component.pre_process_data.get_tokenizer")
    def test_huggingface_pre_process_with_label(self, mock_get_tokenizer, mock_dataset):
        """Test huggingface_pre_process function with label processing."""
        # setup
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[101, 2023, 102], [101, 2178, 102]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        mock_get_tokenizer.return_value = mock_tokenizer

        # execute
        result = huggingface_pre_process(
            dataset=mock_dataset,
            model_name="bert-base-uncased",
            input_cols=["sentence1", "sentence2"],
            label_col="label",
            max_samples=2,
            trust_remote_code=None,
        )

        # verify
        mock_get_tokenizer.assert_called_with("bert-base-uncased", trust_remote_code=None)
        assert result.label_col == "label"
        assert result.effective_len == 2


class TestVisionAndAudioFileName:
    """Tests for the file_name surfaced by vision/audio preprocessors for sample logging."""

    def test_vision_vqa_pre_process_uses_id_col_as_file_name(self):
        data = {
            "image": ["img_a", "img_b"],
            "question": ["Q1", "Q2"],
            "answer": ["1", "2"],
            "image_id": ["a.png", "b.png"],
        }
        dataset = Dataset.from_dict(data)
        vqa = vision_vqa_pre_process(
            dataset, image_col="image", question_col="question", answer_col="answer", id_col="image_id"
        )
        input_dict, answer = vqa[0]
        assert input_dict["file_name"] == "a.png"
        assert input_dict["question"] == "Q1"
        assert answer == "1"

    def test_vision_vqa_pre_process_falls_back_to_index(self):
        # Includes options so the num_choices answer-conversion path runs; file_name must still
        # be the dataset row index (regression: the conversion previously shadowed the index var).
        data = {
            "image": ["img_a", "img_b"],
            "question": ["Q1", "Q2"],
            "answer": ["0", "1"],
            "options": [["a", "b"], ["c", "d"]],
        }
        dataset = Dataset.from_dict(data)
        vqa = vision_vqa_pre_process(
            dataset, image_col="image", question_col="question", answer_col="answer", options_col="options"
        )
        input_dict0, answer0 = vqa[0]
        assert input_dict0["file_name"] == "0"
        assert answer0 == "1"  # 0-based answer converted to 1-based
        input_dict1, _ = vqa[1]
        assert input_dict1["file_name"] == "1"

    @patch("datasets.Dataset.cast_column", autospec=True)
    def test_speech_transcription_pre_process_returns_dict_with_file_name(self, mock_cast):
        import numpy as np

        # cast_column is patched to a no-op so we can supply raw audio dicts directly.
        mock_cast.side_effect = lambda self, *args, **kwargs: self
        data = {
            "audio": [{"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000, "path": "/data/clip_0.wav"}],
            "text": ["hello"],
        }
        dataset = Dataset.from_dict(data)
        speech = speech_transcription_pre_process(dataset, audio_col="audio", text_col="text")
        input_dict, text = speech[0]
        assert input_dict["file_name"] == "clip_0.wav"
        assert input_dict["audio"].shape == (16000,)
        assert text == "hello"
