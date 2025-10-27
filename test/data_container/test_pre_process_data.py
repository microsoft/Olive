# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from olive.data.component.pre_process_data import huggingface_pre_process, tokenizer_pre_process


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
        assert result.label_col is None
        assert result.max_samples == 2

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
        assert result.max_samples == 2
