# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import sys
from unittest.mock import MagicMock, call, patch

import pytest

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
        config.first_n_tokens_timed = 5

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
            result = pass_instance.compare_generation(
                config, mock_ref_model, ref_model_path=config.reference_model_path
            )

        assert mock_generator.append_tokens.call_count == 2
        mock_generator.append_tokens.assert_has_calls([call([[1, 2, 3]]), call([[1, 2, 3]])])
        # Generated-only common prefix: transformers [10, 11, 12, 13] vs genai [10, 11, 99, 99]
        # matches on [10, 11] = 2 tokens before divergence (shared prompt is excluded).
        assert result["longest_common_token_sequence"] == 2
        # Latency metrics are exposed for both transformers and ONNX Runtime GenAI.
        assert result["first_n_tokens_timed"] == 5
        for key in (
            "transformers_ttft_s",
            "transformers_ttfn_s",
        ):
            assert key in result
            assert isinstance(result[key], float)
        for key in ("genai_ttft_s", "genai_ttfn_s"):
            assert key in result

    def test_compare_generation_fully_matching(self):
        """Test when both outputs are identical."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Test"
        config.generate_max_new_tokens = 5
        config.first_n_tokens_timed = 5

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
            result = pass_instance.compare_generation(
                config, mock_ref_model, ref_model_path=config.reference_model_path
            )

        assert mock_generator.append_tokens.call_count == 2
        mock_generator.append_tokens.assert_has_calls([call([[10, 20]]), call([[10, 20]])])
        # All 3 generated tokens match (shared prompt is excluded)
        assert result["longest_common_token_sequence"] == 3
        assert result["first_n_tokens_timed"] == 5
        for key in (
            "transformers_ttft_s",
            "transformers_ttfn_s",
        ):
            assert key in result
            assert isinstance(result[key], float)
        assert "genai_ttft_s" in result
        assert isinstance(result["genai_ttft_s"], float)
        assert "genai_ttfn_s" in result
        assert result["genai_ttfn_s"] is None or isinstance(result["genai_ttfn_s"], float)

    def test_compare_generation_with_zero_max_new_tokens(self):
        """Test that latency metrics are skipped when max_new_tokens is zero."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Test"
        config.generate_max_new_tokens = 0
        config.first_n_tokens_timed = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[10, 20]]))

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[10, 20]])

        mock_og = MagicMock()
        mock_og.Model.return_value = MagicMock()
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [10, 20]
        mock_og.GeneratorParams.return_value = MagicMock()

        mock_generator = MagicMock()
        mock_generator.is_done.return_value = True
        mock_og.Generator.return_value = mock_generator

        with (
            patch.dict(sys.modules, {"onnxruntime_genai": mock_og}),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_generation(
                config, mock_ref_model, ref_model_path=config.reference_model_path
            )

        assert mock_ref_model.generate.call_count == 2
        assert mock_ref_model.generate.call_args.kwargs["max_new_tokens"] == 0
        assert result["first_n_tokens_timed"] == 0
        assert result["transformers_ttft_s"] is None
        assert result["transformers_ttfn_s"] is None

    def test_compare_generation_reports_first_token_match(self):
        """first_token_matches is True when both first generated tokens are identical."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = None
        config.generate_prompt = "Hello world"
        config.generate_max_new_tokens = 10
        config.first_n_tokens_timed = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        # First generated token (after the 3-token prompt) is 10.
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11, 12]])

        mock_og = MagicMock()
        mock_og.Model.return_value = MagicMock()
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [1, 2, 3]
        mock_og.GeneratorParams.return_value = MagicMock()

        mock_generator = MagicMock()
        # GenAI first generated token is also 10 -> match.
        genai_new_tokens = [10, 99, 99]
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
            result = pass_instance.compare_generation(
                config,
                mock_ref_model,
                ref_model_path=config.reference_model_path,
                genai_model_path="explicit_genai_dir",
                max_new_tokens=20,
                first_n=5,
            )

        # The explicit genai_model_path override is used for og.Model.
        mock_og.Model.assert_called_once_with("explicit_genai_dir")
        # The max_new_tokens override is forwarded to transformers.generate.
        assert mock_ref_model.generate.call_args.kwargs["max_new_tokens"] == 20
        assert result["transformers_first_token"] == 10
        assert result["genai_first_token"] == 10
        assert result["first_token_matches"] is True
        # transformers generated [10, 11, 12] and genai [10, 99, 99] -> second tokens differ.
        assert result["transformers_second_token"] == 11
        assert result["genai_second_token"] == 99
        assert result["second_token_matches"] is False

    def test_compare_generation_reports_first_token_mismatch(self):
        """first_token_matches is False when the first generated tokens differ."""
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.genai_model_path = "mock_genai_model"
        config.generate_prompt = "Hello"
        config.generate_max_new_tokens = 10
        config.first_n_tokens_timed = 5

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2]]))

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 30, 31]])

        mock_og = MagicMock()
        mock_og.Model.return_value = MagicMock()
        mock_genai_tokenizer = MagicMock()
        mock_og.Tokenizer.return_value = mock_genai_tokenizer
        mock_genai_tokenizer.encode.return_value = [1, 2]
        mock_og.GeneratorParams.return_value = MagicMock()

        mock_generator = MagicMock()
        genai_new_tokens = [40, 41]
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
            result = pass_instance.compare_generation(
                config, mock_ref_model, ref_model_path=config.reference_model_path
            )

        assert result["transformers_first_token"] == 30
        assert result["genai_first_token"] == 40
        assert result["first_token_matches"] is False
        # transformers generated [30, 31] and genai [40, 41] -> second tokens differ too.
        assert result["transformers_second_token"] == 31
        assert result["genai_second_token"] == 41
        assert result["second_token_matches"] is False


class TestWeightDtypeInference:
    """Unit tests for ONNX weight dtype inference used to match the reference model precision."""

    @staticmethod
    def _build_tiny_llm_onnx(onnx_float_dtype):
        """Build a tiny-llm-like ONNX model whose float weights use the given dtype.

        The graph mimics a minimal language-model head: an embedding weight and a
        linear (lm_head) weight, plus an int64 buffer that must be ignored by the
        float dtype inference.
        """
        from onnx import TensorProto, helper

        hidden = 4
        vocab = 8

        embed_weight = helper.make_tensor("embed.weight", onnx_float_dtype, [vocab, hidden], [0.1] * (vocab * hidden))
        lm_head_weight = helper.make_tensor(
            "lm_head.weight", onnx_float_dtype, [hidden, vocab], [0.2] * (hidden * vocab)
        )
        # Non-float buffer that must be ignored when inferring the weight dtype.
        position_ids = helper.make_tensor("position_ids", TensorProto.INT64, [hidden], [0, 1, 2, 3])

        input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1])
        logits = helper.make_tensor_value_info("logits", onnx_float_dtype, [1, vocab])

        gather = helper.make_node("Gather", ["embed.weight", "input_ids"], ["hidden_states"])
        matmul = helper.make_node("MatMul", ["hidden_states", "lm_head.weight"], ["logits"])

        graph = helper.make_graph(
            [gather, matmul],
            "tiny_llm",
            [input_ids],
            [logits],
            initializer=[embed_weight, lm_head_weight, position_ids],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_infer_onnx_weight_dtype_float32(self):
        import onnx

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype

        model = self._build_tiny_llm_onnx(onnx.TensorProto.FLOAT)
        assert _infer_onnx_weight_dtype(model) == onnx.TensorProto.FLOAT

    def test_infer_onnx_weight_dtype_bfloat16(self):
        import onnx

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype

        model = self._build_tiny_llm_onnx(onnx.TensorProto.BFLOAT16)
        assert _infer_onnx_weight_dtype(model) == onnx.TensorProto.BFLOAT16

    def test_infer_onnx_weight_dtype_returns_none_without_float_weights(self):
        from onnx import TensorProto, helper

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype

        only_int = helper.make_tensor("buffer", TensorProto.INT64, [2], [1, 2])
        graph = helper.make_graph([], "no_float", [], [], initializer=[only_int])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        assert _infer_onnx_weight_dtype(model) is None

    def test_onnx_dtype_to_torch_float32(self):
        import onnx
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_dtype_to_torch

        assert _onnx_dtype_to_torch(onnx.TensorProto.FLOAT) == torch.float32

    def test_onnx_dtype_to_torch_bfloat16(self):
        import onnx
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_dtype_to_torch

        assert _onnx_dtype_to_torch(onnx.TensorProto.BFLOAT16) == torch.bfloat16

    def test_onnx_output_to_torch_bfloat16_from_uint16(self):
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_output_to_torch

        expected = torch.tensor([[1.5, -2.0]], dtype=torch.bfloat16)
        onnx_output = expected.view(torch.uint16).cpu().numpy()
        actual = _onnx_output_to_torch(onnx_output, torch.bfloat16)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, expected)

    def test_onnx_output_to_torch_keeps_uint16_for_integer_reference(self):
        import torch

        from olive.passes.onnx.discrepancy_check import _onnx_output_to_torch

        expected = torch.tensor([[123, 456]], dtype=torch.uint16)
        actual = _onnx_output_to_torch(expected.cpu().numpy(), torch.uint16)
        assert actual.dtype == torch.uint16
        assert torch.equal(actual, expected)

    def test_tiny_llm_reference_model_cast_to_weight_dtype(self):
        """End-to-end check that the reference model is cast to the ONNX weight dtype.

        Builds a tiny-llm ONNX model in float32 and bfloat16, then verifies the
        inferred torch dtype matches what the pass would cast the reference model to.
        """
        import onnx
        import torch

        from olive.passes.onnx.discrepancy_check import _infer_onnx_weight_dtype, _onnx_dtype_to_torch

        for onnx_dtype, expected_torch_dtype in (
            (onnx.TensorProto.FLOAT, torch.float32),
            (onnx.TensorProto.BFLOAT16, torch.bfloat16),
        ):
            model = self._build_tiny_llm_onnx(onnx_dtype)
            inferred = _infer_onnx_weight_dtype(model)
            assert _onnx_dtype_to_torch(inferred) == expected_torch_dtype


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

    def test_measure_speedup_returns_latencies_and_speedup(self):
        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
        ref_model = MagicMock()
        session = MagicMock()
        input_data = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}
        dataloader = [(input_data, None)]

        with (
            patch("olive.common.utils.format_data", return_value={"input_ids": [1, 2, 3]}),
            patch("olive.passes.onnx.discrepancy_check.time.perf_counter", side_effect=[10.0, 14.0, 20.0, 22.0]),
        ):
            result = pass_instance._measure_speedup(
                ref_model=ref_model,
                session=session,
                dataloader=dataloader,
                io_config=MagicMock(),
                torch_device=torch.device("cpu"),
                warmup_iterations=1,
                timing_iterations=2,
            )

        assert result == (2.0, 1.0, 2.0)
        assert ref_model.call_count == 3
        assert session.run.call_count == 3


class TestCompareLlamaCpp:
    """Unit tests for OnnxDiscrepancyCheck.compare_llama_cpp."""

    def _make_config(self):
        config = MagicMock()
        config.reference_model_path = "mock_model"
        config.generate_prompt = "Hello world"
        config.generate_max_new_tokens = 10
        config.first_n_tokens_timed = 5
        config.llama_cpp_env_path = "/mock/llama_env"
        return config

    def _make_hf_config(self):
        hf_cfg = MagicMock()
        hf_cfg.max_position_embeddings = 64
        hf_cfg.hidden_size = 128
        hf_cfg.num_hidden_layers = 2
        hf_cfg.intermediate_size = 256
        hf_cfg.num_attention_heads = 8
        hf_cfg.num_key_value_heads = 8
        hf_cfg.rms_norm_eps = 1e-5
        hf_cfg.vocab_size = 32
        return hf_cfg

    def test_get_llama_env_python_posix(self, tmp_path):
        """Test that the POSIX Python path is returned when it exists."""
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        (tmp_path / "bin").mkdir()
        python = tmp_path / "bin" / "python"
        python.touch()

        result = OnnxDiscrepancyCheck._get_llama_env_python(str(tmp_path))
        assert result == str(python)

    def test_get_llama_env_python_missing_raises(self, tmp_path):
        """Test that a RuntimeError is raised when no interpreter is found."""
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        with pytest.raises(RuntimeError, match="llama_env"):
            OnnxDiscrepancyCheck._get_llama_env_python(str(tmp_path))

    def test_compare_llama_cpp_returns_expected_metrics(self, tmp_path):
        """Test that compare_llama_cpp returns all expected keys and correct values."""
        import json

        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        # First token from transformers: 42
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 42]])
        mock_ref_model.state_dict.return_value = {}

        llama_output = {
            "first_token_id": 42,
            "generated_tokens": [42, 43, 44, 45, 46],
            "ttft": 0.05,
            "ttfn": 0.25,
            "total_time": 0.50,
        }

        mock_proc = MagicMock()
        mock_proc.stdout = json.dumps(llama_output)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            __getitem__=lambda self, key: torch.tensor([[1, 2, 3]]) if key == "input_ids" else None,
        )
        mock_tokenizer.return_value.__getitem__ = lambda self, k: (
            torch.tensor([[1, 2, 3]]) if k == "input_ids" else None
        )
        # tokenizer(prompt) returns a dict with "input_ids" as a list
        encoded = MagicMock()
        encoded.__getitem__ = MagicMock(side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None)
        mock_tokenizer.return_value = encoded
        mock_tokenizer.get_vocab = MagicMock(return_value={})

        with (
            patch.object(OnnxDiscrepancyCheck, "_get_llama_env_python", return_value="/mock/llama_env/bin/python"),
            patch.object(
                OnnxDiscrepancyCheck, "_get_convert_script", return_value="/mock/llama_env/convert_hf_to_gguf.py"
            ),
            patch("subprocess.run", return_value=mock_proc),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoConfig.from_pretrained", return_value=self._make_hf_config()),
            patch("numpy.savez"),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_llama_cpp(
                config,
                mock_ref_model,
                output_dir=str(tmp_path),
                pytorch_latency_s=0.10,
                onnx_latency_s=0.05,
                ref_model_path=config.reference_model_path,
            )

        expected_keys = {
            "llama_cpp_first_token_id",
            "llama_cpp_pytorch_first_token_id",
            "llama_cpp_first_token_matches_pytorch",
            "llama_cpp_second_token_id",
            "llama_cpp_pytorch_second_token_id",
            "llama_cpp_second_token_matches_pytorch",
            "llama_cpp_longest_common_token_sequence",
            "llama_cpp_ttft_s",
            "llama_cpp_ttfn_s",
            "llama_cpp_total_time_s",
        }
        assert expected_keys <= set(result.keys())

        assert result["llama_cpp_first_token_id"] == 42
        # transformers generated only one token ([42]), so there is no reference second token.
        assert result["llama_cpp_pytorch_second_token_id"] is None
        assert result["llama_cpp_second_token_id"] == 43
        assert result["llama_cpp_second_token_matches_pytorch"] is False
        # Generated-only comparison: transformers generated [42] vs llama.cpp [42, 43, ...] = 1 match.
        assert result["llama_cpp_longest_common_token_sequence"] == 1
        assert result["llama_cpp_ttft_s"] == pytest.approx(0.05)
        assert result["llama_cpp_ttfn_s"] == pytest.approx(0.25)
        assert result["llama_cpp_total_time_s"] == pytest.approx(0.50)

    def test_compare_llama_cpp_no_latency_baselines(self, tmp_path):
        """Speedup fields are None when pytorch/onnx latencies are not provided."""
        import json

        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 7]])
        mock_ref_model.state_dict.return_value = {}

        llama_output = {
            "first_token_id": 7,
            "generated_tokens": [7, 8],
            "ttft": 0.10,
            "ttfn": None,
            "total_time": 0.20,
        }

        mock_proc = MagicMock()
        mock_proc.stdout = json.dumps(llama_output)

        encoded = MagicMock()
        encoded.__getitem__ = MagicMock(side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = encoded
        mock_tokenizer.get_vocab = MagicMock(return_value={})

        with (
            patch.object(OnnxDiscrepancyCheck, "_get_llama_env_python", return_value="/mock/llama_env/bin/python"),
            patch.object(
                OnnxDiscrepancyCheck, "_get_convert_script", return_value="/mock/llama_env/convert_hf_to_gguf.py"
            ),
            patch("subprocess.run", return_value=mock_proc),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoConfig.from_pretrained", return_value=self._make_hf_config()),
            patch("numpy.savez"),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_llama_cpp(
                config, mock_ref_model, output_dir=str(tmp_path), ref_model_path=config.reference_model_path
            )

        assert "llama_cpp_speedup_vs_pytorch" not in result
        assert "llama_cpp_speedup_vs_onnx" not in result
        assert result["llama_cpp_first_token_id"] == 7
        assert result["llama_cpp_first_token_matches_pytorch"] is True

    def test_compare_llama_cpp_uses_preconverted_gguf(self, tmp_path):
        import json

        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()
        gguf_path = tmp_path / "prebuilt.gguf"
        gguf_path.write_text("ok")

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 7]])

        llama_output = {
            "first_token_id": 7,
            "generated_tokens": [7, 8],
            "ttft": 0.10,
            "ttfn": None,
            "total_time": 0.20,
        }

        mock_proc = MagicMock()
        mock_proc.stdout = json.dumps(llama_output)

        encoded = MagicMock()
        encoded.__getitem__ = MagicMock(side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = encoded
        mock_tokenizer.get_vocab = MagicMock(return_value={})

        with (
            patch.object(OnnxDiscrepancyCheck, "_get_llama_env_python", return_value="/mock/llama_env/bin/python"),
            patch.object(OnnxDiscrepancyCheck, "_get_convert_script") as mock_convert_script,
            patch("subprocess.run", return_value=mock_proc) as mock_subprocess_run,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_llama_cpp(
                config,
                mock_ref_model,
                output_dir=str(tmp_path),
                ref_model_path=config.reference_model_path,
                preconverted_gguf_path=str(gguf_path),
            )

        assert result["llama_cpp_first_token_id"] == 7
        mock_convert_script.assert_not_called()
        assert mock_subprocess_run.call_count == 1

    def test_compare_llama_cpp_returns_stderr_and_stdout_on_helper_error(self, tmp_path):
        import subprocess

        import torch

        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()
        gguf_path = tmp_path / "prebuilt.gguf"
        gguf_path.write_text("ok")

        mock_ref_model = MagicMock()
        mock_ref_model.device = torch.device("cpu")
        mock_ref_model.generate.return_value = torch.tensor([[1, 2, 3, 7]])

        encoded = MagicMock()
        encoded.__getitem__ = MagicMock(side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = encoded
        mock_tokenizer.get_vocab = MagicMock(return_value={})

        helper_error = subprocess.CalledProcessError(
            1, ["python", "llama_cpp_helper.py"], output="stdout text", stderr="stderr text"
        )

        with (
            patch.object(OnnxDiscrepancyCheck, "_get_llama_env_python", return_value="/mock/llama_env/bin/python"),
            patch.object(OnnxDiscrepancyCheck, "_get_convert_script"),
            patch("subprocess.run", side_effect=helper_error),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
            result = pass_instance.compare_llama_cpp(
                config,
                mock_ref_model,
                output_dir=str(tmp_path),
                ref_model_path=config.reference_model_path,
                preconverted_gguf_path=str(gguf_path),
            )

        assert result == {"llama_cpp_out": "stderr text", "llama_cpp_err": "stdout text"}


class TestMobiusExporterGenerationComparison:
    """OnnxDiscrepancyCheck must accept a MobiusBuilder-produced model as a GenAI exporter.

    MobiusBuilder (like ModelBuilder) emits ORT GenAI artifacts (``genai_config.json``) alongside
    the ONNX model. The generation comparison auto-detects that directory and uses it as the GenAI
    model, so a mobius-exported model must drive the transformers-vs-GenAI comparison exactly like a
    model-builder-exported one.
    """

    def _make_config(self):
        config = MagicMock()
        config.genai_model_path = None
        config.generate_max_new_tokens = 10
        config.first_n_tokens_timed = 5
        config.min_longest_common_tokens = None
        return config

    def _make_mobius_output(self, tmp_path):
        """Create a single-component MobiusBuilder output directory (model.onnx + genai_config.json)."""
        import json

        (tmp_path / "model.onnx").write_bytes(b"onnx")
        (tmp_path / "genai_config.json").write_text(json.dumps({"model": {"type": "llama"}}))

        model = MagicMock()
        model.model_path = str(tmp_path)
        # Attributes stamped by MobiusBuilder for a single-component (LLM) export.
        model.model_attributes = {"mobius_package_keys": ["decoder"]}
        return model

    def test_mobius_output_is_used_as_genai_model(self, tmp_path):
        """A mobius export dir with genai_config.json is auto-detected and passed to compare_generation."""
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()
        model = self._make_mobius_output(tmp_path)
        ref_model = MagicMock()

        gen_results = {
            "longest_common_token_sequence": 6,
            "first_n_tokens_timed": 5,
            "transformers_first_token": 100,
            "genai_first_token": 100,
            "first_token_matches": True,
            "transformers_second_token": 200,
            "genai_second_token": 200,
            "second_token_matches": True,
            "transformers_ttft_s": 0.01,
            "transformers_ttfn_s": 0.02,
            "genai_ttft_s": 0.03,
            "genai_ttfn_s": 0.04,
        }
        results = {"status": "passed"}

        pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
        with patch.object(OnnxDiscrepancyCheck, "compare_generation", return_value=gen_results) as mock_compare:
            pass_instance._run_generation_comparison(
                model, config, ref_model, "ref_path", {"first_token_20"}, results
            )

        # The mobius output directory was resolved as the GenAI model and passed through.
        mock_compare.assert_called_once()
        _, kwargs = mock_compare.call_args
        assert kwargs["genai_model_path"] == str(tmp_path)
        assert results["genai_model_path"] == str(tmp_path)
        # Generation metrics from the mobius-exported model are surfaced.
        assert results["first_token_20"]["first_token_matches"] is True
        assert results["longest_common_token_sequence"] == 6

    def test_mobius_first_token_20_requests_20_tokens(self, tmp_path):
        """first_token_20 must request a 20-token generation from the mobius-exported GenAI model."""
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()
        model = self._make_mobius_output(tmp_path)

        gen_results = {
            "longest_common_token_sequence": 3,
            "first_n_tokens_timed": 5,
            "transformers_first_token": 1,
            "genai_first_token": 1,
            "first_token_matches": True,
            "transformers_second_token": 2,
            "genai_second_token": 2,
            "second_token_matches": True,
            "transformers_ttft_s": None,
            "transformers_ttfn_s": None,
            "genai_ttft_s": None,
            "genai_ttfn_s": None,
        }
        results = {"status": "passed"}

        pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
        with patch.object(OnnxDiscrepancyCheck, "compare_generation", return_value=gen_results) as mock_compare:
            pass_instance._run_generation_comparison(
                model, config, MagicMock(), "ref_path", {"first_token_20"}, results
            )

        _, kwargs = mock_compare.call_args
        assert kwargs["max_new_tokens"] == 20

    def test_mobius_output_without_genai_config_is_skipped(self, tmp_path):
        """Without genai_config.json the mobius dir is not treated as a GenAI model (no comparison)."""
        from olive.passes.onnx.discrepancy_check import OnnxDiscrepancyCheck

        config = self._make_config()
        # ONNX model present but no genai_config.json emitted.
        (tmp_path / "model.onnx").write_bytes(b"onnx")
        model = MagicMock()
        model.model_path = str(tmp_path)
        model.model_attributes = {"mobius_package_keys": ["decoder"]}
        results = {"status": "passed"}

        pass_instance = OnnxDiscrepancyCheck.__new__(OnnxDiscrepancyCheck)
        with patch.object(OnnxDiscrepancyCheck, "compare_generation") as mock_compare:
            pass_instance._run_generation_comparison(
                model, config, MagicMock(), "ref_path", {"first_token_20"}, results
            )

        mock_compare.assert_not_called()
        assert "genai_model_path" not in results
