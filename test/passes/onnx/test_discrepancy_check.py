# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import sys
from unittest.mock import MagicMock, patch

from olive.passes.onnx.discrepancy_check import _longest_common_token_sequence


class TestLongestCommonTokenSequence:
    """Unit tests for _longest_common_token_sequence helper."""

    def test_identical_sequences(self):
        assert _longest_common_token_sequence([1, 2, 3, 4], [1, 2, 3, 4]) == 4

    def test_empty_sequences(self):
        assert _longest_common_token_sequence([], []) == 0
        assert _longest_common_token_sequence([1, 2], []) == 0
        assert _longest_common_token_sequence([], [1, 2]) == 0

    def test_no_common_prefix(self):
        assert _longest_common_token_sequence([1, 2, 3], [4, 5, 6]) == 0

    def test_partial_common_prefix(self):
        assert _longest_common_token_sequence([1, 2, 3, 4, 5], [1, 2, 3, 9, 10]) == 3

    def test_one_is_prefix_of_other(self):
        assert _longest_common_token_sequence([1, 2, 3], [1, 2, 3, 4, 5]) == 3
        assert _longest_common_token_sequence([1, 2, 3, 4, 5], [1, 2, 3]) == 3

    def test_single_element_match(self):
        assert _longest_common_token_sequence([7], [7]) == 1

    def test_single_element_no_match(self):
        assert _longest_common_token_sequence([7], [8]) == 0

    def test_diverges_after_first(self):
        assert _longest_common_token_sequence([1, 99, 99], [1, 2, 3]) == 1

    def test_common_tokens_later_not_counted(self):
        # Tokens match later but not from the beginning
        assert _longest_common_token_sequence([10, 1, 2, 3], [20, 1, 2, 3]) == 0


class TestCompareGeneration:
    """Unit tests for OnnxDiscrepancyCheck.compare_generation."""

    def test_compare_generation_returns_common_prefix_length(self):
        """Test that compare_generation correctly computes the common prefix length."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        # Mock config
        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Hello world"
        config.generate_max_new_tokens = 10

        # Mock transformers tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        # Transformers generates [1, 2, 3, 10, 11, 12, 13]
        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11, 12, 13]])

        # GenAI generates [1, 2, 3, 10, 11, 99, 99] (diverges at index 5)
        mock_og = MagicMock()
        mock_genai_model = MagicMock()
        mock_og.Model.return_value = mock_genai_model
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [1, 2, 3]

        mock_params = MagicMock()
        mock_og.GeneratorParams.return_value = mock_params

        # Simulate generator producing tokens: 10, 11, 99, 99 then done
        mock_generator = MagicMock()
        genai_new_tokens = [10, 11, 99, 99]
        call_count = [0]

        def is_done_side_effect():
            return call_count[0] >= len(genai_new_tokens)

        def get_next_tokens_side_effect():
            token = genai_new_tokens[call_count[0]]
            call_count[0] += 1
            return [token]

        mock_generator.is_done = is_done_side_effect
        mock_generator.get_next_tokens = get_next_tokens_side_effect
        mock_og.Generator.return_value = mock_generator

        with (
            patch.dict(sys.modules, {"onnxruntime_genai": mock_og}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_generation(config, mock_ref_model)

        mock_generator.append_tokens.assert_called_once_with([[1, 2, 3]])
        # Common prefix: [1, 2, 3, 10, 11] = 5 tokens before divergence
        assert result == 5

    def test_compare_generation_fully_matching(self):
        """Test when both outputs are identical."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Test"
        config.generate_max_new_tokens = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[10, 20]]))

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50]])

        mock_og = MagicMock()
        mock_og.Model.return_value = MagicMock()
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [10, 20]

        mock_params = MagicMock()
        mock_og.GeneratorParams.return_value = mock_params

        mock_generator = MagicMock()
        genai_new_tokens = [30, 40, 50]
        call_count = [0]

        def is_done_side_effect():
            return call_count[0] >= len(genai_new_tokens)

        def get_next_tokens_side_effect():
            token = genai_new_tokens[call_count[0]]
            call_count[0] += 1
            return [token]

        mock_generator.is_done = is_done_side_effect
        mock_generator.get_next_tokens = get_next_tokens_side_effect
        mock_og.Generator.return_value = mock_generator

        with (
            patch.dict(sys.modules, {"onnxruntime_genai": mock_og}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_generation(config, mock_ref_model)

        mock_generator.append_tokens.assert_called_once_with([[10, 20]])
        # All 5 tokens match
        assert result == 5


class TestSpeedupSettings:
    def test_timing_iterations_default_is_5(self):
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        pass_config = OnnxDiscrepancyCheck._default_config(None)
        assert pass_config["timing_iterations"].default_value == 5

    def test_measure_speedup_skips_when_timing_iterations_is_zero(self):
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
        ref_model = MagicMock()
        session = MagicMock()

        result = pass_instance._measure_speedup(
            ref_model=ref_model,
            session=session,
            dataloader=MagicMock(),
            io_config=MagicMock(),
            torch_device=MagicMock(),
            warmup_iterations=3,
            timing_iterations=0,
        )

        assert result is None
        ref_model.assert_not_called()
        session.run.assert_not_called()
