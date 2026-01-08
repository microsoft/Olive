# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

import inspect
import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar

from olive.common.hf.io_config.input_generators import DummyInputGenerator
from olive.common.hf.io_config.normalized_config import NormalizedConfig, NormalizedConfigManager
from olive.common.hf.io_config.tasks import TaskType

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class OnnxConfig(ABC):
    """Base class for ONNX exportable model configurations."""

    DEFAULT_ONNX_OPSET = 18
    NORMALIZED_CONFIG_CLASS: ClassVar[type] = NormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES: ClassVar[tuple] = ()

    _TASK_TO_COMMON_OUTPUTS: ClassVar[dict] = {
        TaskType.AUDIO_CLASSIFICATION: OrderedDict({"logits": {0: "batch_size"}}),
        TaskType.AUDIO_FRAME_CLASSIFICATION: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.AUTOMATIC_SPEECH_RECOGNITION: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.AUDIO_XVECTOR: OrderedDict({"logits": {0: "batch_size"}, "embeddings": {0: "batch_size"}}),
        TaskType.DEPTH_ESTIMATION: OrderedDict({"predicted_depth": {0: "batch_size", 1: "height", 2: "width"}}),
        TaskType.DOCUMENT_QUESTION_ANSWERING: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.FEATURE_EXTRACTION: OrderedDict({"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.FILL_MASK: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.IMAGE_CLASSIFICATION: OrderedDict({"logits": {0: "batch_size"}}),
        TaskType.IMAGE_SEGMENTATION: OrderedDict({"logits": {0: "batch_size", 2: "height", 3: "width"}}),
        TaskType.IMAGE_TO_TEXT: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.IMAGE_TO_IMAGE: OrderedDict(
            {"reconstruction": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}
        ),
        TaskType.KEYPOINT_DETECTION: OrderedDict(
            {"heatmaps": {0: "batch_size", 1: "num_keypoints", 2: "height", 3: "width"}}
        ),
        TaskType.MASK_GENERATION: OrderedDict({"logits": {0: "batch_size"}}),
        TaskType.MASKED_IM: OrderedDict({"reconstruction": {0: "batch_size"}}),
        TaskType.MULTIPLE_CHOICE: OrderedDict({"logits": {0: "batch_size", 1: "num_choices"}}),
        TaskType.OBJECT_DETECTION: OrderedDict(
            {
                "logits": {0: "batch_size", 1: "num_queries"},
                "pred_boxes": {0: "batch_size", 1: "num_queries"},
            }
        ),
        TaskType.QUESTION_ANSWERING: OrderedDict(
            {
                "start_logits": {0: "batch_size", 1: "sequence_length"},
                "end_logits": {0: "batch_size", 1: "sequence_length"},
            }
        ),
        TaskType.SEMANTIC_SEGMENTATION: OrderedDict(
            {"logits": {0: "batch_size", 1: "num_labels", 2: "height", 3: "width"}}
        ),
        TaskType.TEXT2TEXT_GENERATION: OrderedDict({"logits": {0: "batch_size", 1: "decoder_sequence_length"}}),
        TaskType.TEXT_CLASSIFICATION: OrderedDict({"logits": {0: "batch_size"}}),
        TaskType.TEXT_GENERATION: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.TIME_SERIES_FORECASTING: OrderedDict({"prediction_outputs": {0: "batch_size"}}),
        TaskType.TOKEN_CLASSIFICATION: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.VISUAL_QUESTION_ANSWERING: OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION: OrderedDict(
            {
                "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
                "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        ),
        TaskType.ZERO_SHOT_OBJECT_DETECTION: OrderedDict(
            {
                "logits": {0: "batch_size", 1: "num_queries"},
                "pred_boxes": {0: "batch_size", 1: "num_queries"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        ),
    }

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        self._config = config
        self.task = task
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

        # Initialize normalized config
        if self.NORMALIZED_CONFIG_CLASS is not None:
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(config)
        else:
            # Fallback to NormalizedConfigManager
            model_type = getattr(config, "model_type", None)
            if model_type:
                normalized_config_class = NormalizedConfigManager.get_normalized_config_class(model_type)
                self._normalized_config = normalized_config_class(config)
            else:
                self._normalized_config = NormalizedConfig(config)

    @property
    @abstractmethod
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return the inputs of the model."""
        raise NotImplementedError

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return the outputs of the model."""
        # Return a copy to avoid mutating the shared _TASK_TO_COMMON_OUTPUTS
        return OrderedDict(self._TASK_TO_COMMON_OUTPUTS.get(self.task, OrderedDict()))

    def ordered_inputs(self, model: PreTrainedModel) -> dict[str, dict[int, str]]:
        """Re-order the inputs using the model forward pass signature."""
        inputs = self.inputs

        ordered_inputs = {}
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.call)

        for param in sig.parameters:
            param_regex = re.compile(rf"{param}(\..*)?$")
            ordered_inputs.update(
                {name: dynamic_axes for name, dynamic_axes in inputs.items() if re.match(param_regex, name)}
            )
        return ordered_inputs

    def get_io_config(self, model: PreTrainedModel) -> dict[str, Any]:
        """Get complete IO config for ONNX export.

        Args:
            model: The model to get IO config for.

        Returns:
            A dict containing input_names, output_names, dynamic_axes, and dynamic_shapes.

        """
        inputs = self.ordered_inputs(model)
        outputs = self.outputs
        return {
            "input_names": list(inputs.keys()),
            "output_names": list(outputs.keys()),
            "dynamic_axes": {**inputs, **outputs},
            "dynamic_shapes": self._unflatten_past_key_values(inputs),
        }

    @staticmethod
    def _unflatten_past_key_values(flattened_inputs: dict[str, Any]) -> dict[str, Any]:
        """Convert flattened past_key_values to nested format for dynamic_shapes.

        Converts: {"past_key_values.0.key": ..., "past_key_values.0.value": ...}
        To: {"past_key_values": [[key_shape, value_shape], ...]}
        """
        max_idx = -1
        past_key_value_count = 0

        # Find the max index and count past_key_values entries
        for input_name in flattened_inputs:
            if input_name.startswith("past_key_values"):
                idx = int(input_name.split(".")[1])
                max_idx = max(max_idx, idx)
                past_key_value_count += 1

        # Validate count
        expected_count = 2 * (max_idx + 1)
        if past_key_value_count != expected_count or past_key_value_count % 2 != 0:
            logger.warning(
                "Expected %d past_key_values entries, but found %d. "
                "Giving up generating dynamic_shapes. Olive will use dynamic_axes instead.",
                expected_count,
                past_key_value_count,
            )
            return {}

        # No past_key_values found
        if max_idx == -1:
            return flattened_inputs

        # Keep all inputs except past_key_values
        unflattened = {
            input_name: dynamic_shapes
            for input_name, dynamic_shapes in flattened_inputs.items()
            if not input_name.startswith("past_key_values")
        }

        # Generate nested past_key_values list
        unflattened["past_key_values"] = [
            [flattened_inputs[f"past_key_values.{idx}.key"], flattened_inputs[f"past_key_values.{idx}.value"]]
            for idx in range(max_idx + 1)
        ]
        return unflattened

    def _create_dummy_input_generator_classes(self, **kwargs) -> list[DummyInputGenerator]:
        """Create the dummy input generator instances."""
        return [cls(self.task, self._normalized_config, **kwargs) for cls in self.DUMMY_INPUT_GENERATOR_CLASSES if cls]

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> dict[str, Any]:
        """Generate dummy inputs for tracing the model."""
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = list(self.inputs.keys())

        for input_name in input_names:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(
                        input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". '
                    "Try adding a proper dummy input generator to the model ONNX config."
                )

        return dummy_inputs


class OnnxConfigWithPast(OnnxConfig, ABC):
    """A base class to handle the ONNX configuration of decoder-only models with past key values."""

    PAD_ATTENTION_MASK_TO_PAST: bool = False
    SUPPORTS_PAST: bool = True

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
    ):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype)
        self.use_past = use_past
        self.use_past_in_inputs = use_past_in_inputs
        self.is_merged = False
        self.use_cache_branch = None

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if not self.use_past_in_inputs:
            common_outputs = super().outputs
        elif self.task == TaskType.FEATURE_EXTRACTION:
            common_outputs = OrderedDict({"last_hidden_state": {0: "batch_size"}})
        else:
            common_outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}})
        if self.use_past:
            self.add_past_key_values(common_outputs, direction="outputs")
        return common_outputs

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        """Fill input_or_outputs mapping with past_key_values dynamic axes."""
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> dict[str, Any]:
        """Generate dummy inputs for tracing the model with past key values support."""
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs if not key.startswith("past_key_values")]
        if self.use_past_in_inputs and self.use_cache_branch is not False:
            input_names.append("past_key_values")

        for input_name in input_names:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                        dummy_input_gen,
                        input_name,
                        framework,
                        input_shapes=kwargs,
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". '
                    "Try adding a proper dummy input generator to the model ONNX config."
                )

        # Pad attention mask for past key values
        if (
            self.use_past_in_inputs
            and self.PAD_ATTENTION_MASK_TO_PAST
            and self.use_cache_branch is not False
            and "attention_mask" in dummy_inputs
            and self.task == TaskType.TEXT_GENERATION
        ):
            seq_len = dummy_inputs["input_ids"].shape[1]
            past_seq_len = dummy_inputs["past_key_values"][0][1].shape[-2]
            dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["attention_mask"], desired_length=past_seq_len + seq_len, dim=1
            )

        return dummy_inputs

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: DummyInputGenerator, input_name: str, framework: str, input_shapes: dict
    ):
        """Overwrite some shapes and generate the dummy input."""
        if (
            self.use_past
            and self.use_past_in_inputs
            and self.use_cache_branch is not False
            and input_name in ["decoder_input_ids", "input_ids", "position_ids"]
            and self.task != TaskType.TEXT_GENERATION
        ):
            sequence_length = dummy_input_gen.sequence_length
            dummy_input_gen.sequence_length = 1
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )
            dummy_input_gen.sequence_length = sequence_length
        else:
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )

        return dummy_input
