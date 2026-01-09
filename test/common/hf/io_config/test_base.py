# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.common.hf.io_config.base import OnnxConfig, OnnxConfigWithPast
from olive.common.hf.io_config.input_generators import DummyTextInputGenerator
from olive.common.hf.io_config.normalized_config import NormalizedTextConfig
from olive.common.hf.io_config.tasks import TaskType

# ruff: noqa: SLF001


class ConcreteOnnxConfig(OnnxConfig):
    """Concrete implementation for testing."""

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }


class ConcreteOnnxConfigWithPast(OnnxConfigWithPast):
    """Concrete implementation with past for testing."""

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self):
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs


class TestOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        config.model_type = "bert"
        return config

    def test_default_opset(self):
        """Test DEFAULT_ONNX_OPSET is set."""
        assert OnnxConfig.DEFAULT_ONNX_OPSET == 18

    def test_init_sets_task(self, mock_config):
        """Test initialization sets task correctly."""
        onnx_config = ConcreteOnnxConfig(mock_config, task="text-classification")
        assert onnx_config.task == "text-classification"

    def test_init_sets_dtypes(self, mock_config):
        """Test initialization sets dtypes correctly."""
        onnx_config = ConcreteOnnxConfig(mock_config, int_dtype="int32", float_dtype="fp16")
        assert onnx_config.int_dtype == "int32"
        assert onnx_config.float_dtype == "fp16"

    def test_outputs_returns_task_outputs(self, mock_config):
        """Test outputs returns task-specific outputs."""
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.TEXT_CLASSIFICATION)
        outputs = onnx_config.outputs
        assert "logits" in outputs

    def test_outputs_returns_copy(self, mock_config):
        """Test outputs returns a copy to avoid mutation."""
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.TEXT_CLASSIFICATION)
        outputs1 = onnx_config.outputs
        outputs2 = onnx_config.outputs
        assert outputs1 is not outputs2

    def test_ordered_inputs_uses_model_signature(self, mock_config):
        """Test ordered_inputs orders inputs based on model.forward signature."""
        mock_model = MagicMock()

        def forward(input_ids, attention_mask):
            pass

        mock_model.forward = forward
        onnx_config = ConcreteOnnxConfig(mock_config)
        ordered = onnx_config.ordered_inputs(mock_model)
        keys = list(ordered.keys())
        assert keys[0] == "input_ids"
        assert keys[1] == "attention_mask"

    def test_get_io_config(self, mock_config):
        """Test get_io_config returns complete IO config."""
        mock_model = MagicMock()

        def forward(input_ids, attention_mask):
            pass

        mock_model.forward = forward
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.TEXT_CLASSIFICATION)
        io_config = onnx_config.get_io_config(mock_model)

        assert "input_names" in io_config
        assert "output_names" in io_config
        assert "dynamic_axes" in io_config
        assert "dynamic_shapes" in io_config
        assert "input_ids" in io_config["input_names"]
        assert "logits" in io_config["output_names"]

    def test_generate_dummy_inputs(self, mock_config):
        """Test generate_dummy_inputs creates inputs for all input names."""
        onnx_config = ConcreteOnnxConfig(mock_config)
        dummy_inputs = onnx_config.generate_dummy_inputs()

        assert "input_ids" in dummy_inputs
        assert "attention_mask" in dummy_inputs
        assert dummy_inputs["input_ids"].shape[0] == 2  # default batch_size

    def test_generate_dummy_inputs_raises_for_unsupported(self, mock_config):
        """Test generate_dummy_inputs raises for unsupported input names."""

        class BadConfig(OnnxConfig):
            NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
            DUMMY_INPUT_GENERATOR_CLASSES = ()

            @property
            def inputs(self):
                return {"unsupported_input": {0: "batch_size"}}

        onnx_config = BadConfig(mock_config)
        with pytest.raises(RuntimeError, match="Could not generate dummy input"):
            onnx_config.generate_dummy_inputs()


class TestOnnxConfigUnflattenPastKeyValues:
    def test_unflatten_no_past_key_values(self):
        """Test unflatten with no past_key_values returns input as-is."""
        inputs = {"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}}
        result = OnnxConfig._unflatten_past_key_values(inputs)
        assert result == inputs

    def test_unflatten_with_past_key_values(self):
        """Test unflatten converts flattened past_key_values to nested."""
        inputs = {
            "input_ids": {0: "batch"},
            "past_key_values.0.key": {0: "batch", 2: "seq"},
            "past_key_values.0.value": {0: "batch", 2: "seq"},
            "past_key_values.1.key": {0: "batch", 2: "seq"},
            "past_key_values.1.value": {0: "batch", 2: "seq"},
        }
        result = OnnxConfig._unflatten_past_key_values(inputs)

        assert "past_key_values" in result
        assert len(result["past_key_values"]) == 2
        assert result["past_key_values"][0] == [{0: "batch", 2: "seq"}, {0: "batch", 2: "seq"}]

    def test_unflatten_invalid_count_returns_empty(self):
        """Test unflatten returns empty dict for invalid past_key_values count."""
        inputs = {
            "past_key_values.0.key": {0: "batch"},
            # Missing value - odd count
        }
        result = OnnxConfig._unflatten_past_key_values(inputs)
        assert result == {}


class TestOnnxConfigWithPast:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    def test_init_with_use_past(self, mock_config):
        """Test initialization with use_past."""
        onnx_config = ConcreteOnnxConfigWithPast(mock_config, use_past=True, use_past_in_inputs=True)
        assert onnx_config.use_past is True
        assert onnx_config.use_past_in_inputs is True

    def test_outputs_includes_present_when_use_past(self, mock_config):
        """Test outputs include present key values when use_past is True."""
        onnx_config = ConcreteOnnxConfigWithPast(mock_config, task=TaskType.TEXT_GENERATION, use_past=True)
        outputs = onnx_config.outputs
        # Should have present.*.key and present.*.value entries
        present_keys = [k for k in outputs if k.startswith("present")]
        assert len(present_keys) > 0

    def test_add_past_key_values_inputs(self, mock_config):
        """Test add_past_key_values adds past entries for inputs."""
        onnx_config = ConcreteOnnxConfigWithPast(mock_config, use_past=True)
        inputs = {}
        onnx_config.add_past_key_values(inputs, direction="inputs")

        assert "past_key_values.0.key" in inputs
        assert "past_key_values.0.value" in inputs
        assert inputs["past_key_values.0.key"][2] == "past_sequence_length"

    def test_add_past_key_values_outputs(self, mock_config):
        """Test add_past_key_values adds present entries for outputs."""
        onnx_config = ConcreteOnnxConfigWithPast(mock_config, use_past=True)
        outputs = {}
        onnx_config.add_past_key_values(outputs, direction="outputs")

        assert "present.0.key" in outputs
        assert "present.0.value" in outputs
        assert outputs["present.0.key"][2] == "past_sequence_length + sequence_length"

    def test_add_past_key_values_invalid_direction_raises(self, mock_config):
        """Test add_past_key_values raises for invalid direction."""
        onnx_config = ConcreteOnnxConfigWithPast(mock_config)
        with pytest.raises(ValueError, match="direction must either be"):
            onnx_config.add_past_key_values({}, direction="invalid")


class TestOnnxConfigWithPastOutputs:
    """Test OnnxConfigWithPast.outputs variations."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    def test_outputs_with_use_past_in_inputs_feature_extraction(self, mock_config):
        """Test outputs with use_past_in_inputs=True for FEATURE_EXTRACTION."""
        onnx_config = ConcreteOnnxConfigWithPast(
            mock_config,
            task=TaskType.FEATURE_EXTRACTION,
            use_past=True,
            use_past_in_inputs=True,
        )
        outputs = onnx_config.outputs
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"] == {0: "batch_size"}

    def test_outputs_with_use_past_in_inputs_other_task(self, mock_config):
        """Test outputs with use_past_in_inputs=True for non-feature-extraction task."""
        onnx_config = ConcreteOnnxConfigWithPast(
            mock_config,
            task=TaskType.TEXT_CLASSIFICATION,
            use_past=True,
            use_past_in_inputs=True,
        )
        outputs = onnx_config.outputs
        assert "logits" in outputs

    def test_outputs_is_merged_true(self, mock_config):
        """Test outputs when is_merged is True."""
        from olive.common.hf.io_config.config import TextDecoderOnnxConfig

        onnx_config = TextDecoderOnnxConfig(
            mock_config,
            task=TaskType.TEXT_GENERATION,
            use_past=True,
        )
        onnx_config.is_merged = True
        outputs = onnx_config.outputs
        assert "logits" in outputs
        # When merged, sequence_length should be variable
        assert outputs["logits"][1] == "sequence_length"


class TestOnnxConfigWithPastGenerateDummyInputs:
    """Test OnnxConfigWithPast.generate_dummy_inputs."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 4
        config.num_attention_heads = 12
        return config

    def test_generate_dummy_inputs_with_past(self, mock_config):
        """Test generate_dummy_inputs with past key values."""
        from olive.common.hf.io_config.config import TextDecoderOnnxConfig

        onnx_config = TextDecoderOnnxConfig(
            mock_config,
            task=TaskType.TEXT_GENERATION,
            use_past=True,
            use_past_in_inputs=True,
        )
        dummy_inputs = onnx_config.generate_dummy_inputs()
        assert "input_ids" in dummy_inputs
        assert "attention_mask" in dummy_inputs
        assert "past_key_values" in dummy_inputs

    def test_generate_dummy_inputs_use_cache_branch_false(self, mock_config):
        """Test generate_dummy_inputs when use_cache_branch is False."""
        from olive.common.hf.io_config.config import TextDecoderOnnxConfig

        onnx_config = TextDecoderOnnxConfig(
            mock_config,
            task=TaskType.TEXT_GENERATION,
            use_past=True,
            use_past_in_inputs=True,
        )
        onnx_config.use_cache_branch = False
        dummy_inputs = onnx_config.generate_dummy_inputs()
        # past_key_values should not be in inputs when use_cache_branch is False
        assert "past_key_values" not in dummy_inputs


class TestOnnxConfigFallbacks:
    """Test OnnxConfig fallback behavior."""

    def test_ordered_inputs_uses_call_method(self):
        """Test ordered_inputs uses model.call when no forward method."""
        mock_config = MagicMock()
        mock_config.vocab_size = 32000
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12

        mock_model = MagicMock(spec=[])  # No forward attribute
        del mock_model.forward  # Ensure no forward

        def call(input_ids, attention_mask):
            pass

        mock_model.call = call

        onnx_config = ConcreteOnnxConfig(mock_config)
        ordered = onnx_config.ordered_inputs(mock_model)
        assert "input_ids" in ordered

    def test_normalized_config_fallback_with_model_type(self):
        """Test NORMALIZED_CONFIG_CLASS=None falls back to NormalizedConfigManager."""

        class ConfigWithNoNormalizedClass(OnnxConfig):
            NORMALIZED_CONFIG_CLASS = None
            DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

            @property
            def inputs(self):
                return {"input_ids": {0: "batch_size"}}

        mock_config = MagicMock()
        mock_config.vocab_size = 32000
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.model_type = "bert"

        onnx_config = ConfigWithNoNormalizedClass(mock_config)
        assert onnx_config._normalized_config is not None

    def test_normalized_config_fallback_without_model_type(self):
        """Test NORMALIZED_CONFIG_CLASS=None falls back to NormalizedConfig when no model_type."""
        from olive.common.hf.io_config.normalized_config import NormalizedConfig

        class ConfigWithNoNormalizedClass(OnnxConfig):
            NORMALIZED_CONFIG_CLASS = None
            DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

            @property
            def inputs(self):
                return {"input_ids": {0: "batch_size"}}

        mock_config = MagicMock()
        mock_config.vocab_size = 32000
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.model_type = None

        onnx_config = ConfigWithNoNormalizedClass(mock_config)
        assert isinstance(onnx_config._normalized_config, NormalizedConfig)


class TestTaskToCommonOutputs:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    def test_text_classification_outputs(self, mock_config):
        """Test TEXT_CLASSIFICATION outputs."""
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.TEXT_CLASSIFICATION)
        outputs = onnx_config.outputs
        assert "logits" in outputs
        assert outputs["logits"] == {0: "batch_size"}

    def test_question_answering_outputs(self, mock_config):
        """Test QUESTION_ANSWERING outputs."""
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.QUESTION_ANSWERING)
        outputs = onnx_config.outputs
        assert "start_logits" in outputs
        assert "end_logits" in outputs

    def test_feature_extraction_outputs(self, mock_config):
        """Test FEATURE_EXTRACTION outputs."""
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.FEATURE_EXTRACTION)
        outputs = onnx_config.outputs
        assert "last_hidden_state" in outputs

    def test_object_detection_outputs(self, mock_config):
        """Test OBJECT_DETECTION outputs."""
        onnx_config = ConcreteOnnxConfig(mock_config, task=TaskType.OBJECT_DETECTION)
        outputs = onnx_config.outputs
        assert "logits" in outputs
        assert "pred_boxes" in outputs
