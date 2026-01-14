# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from olive.common.hf.io_config.base import OnnxConfig, OnnxConfigWithPast
from olive.common.hf.io_config.input_generators import (
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
)
from olive.common.hf.io_config.tasks import TaskType

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class TextEncoderOnnxConfig(OnnxConfig):
    """Handles encoder-based text architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)


class TextDecoderOnnxConfig(OnnxConfigWithPast):
    """Handles decoder-based text architectures."""

    PAD_ATTENTION_MASK_TO_PAST = True
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
        )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.use_past_in_inputs:
            common_inputs = {"input_ids": {0: "batch_size", 1: "sequence_length"}}
            common_inputs["attention_mask"] = {0: "batch_size", 1: "past_sequence_length + sequence_length"}
            self.add_past_key_values(common_inputs, direction="inputs")
        else:
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }
        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.is_merged is False:
            common_outputs = super().outputs
        else:
            # in the merged case, we need to allow the `sequence_length` to be variable
            common_outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}})
            self.add_past_key_values(common_outputs, direction="outputs")
        return common_outputs


class TextDecoderWithPositionIdsOnnxConfig(TextDecoderOnnxConfig):
    """Handles decoder-based text architectures with position_ids input."""

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = super().inputs

        # Decoders based on GPT2 require a position_ids input
        if self.task in {TaskType.TEXT_GENERATION, TaskType.FEATURE_EXTRACTION}:
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs


class TextSeq2SeqOnnxConfig(OnnxConfigWithPast):
    """Handles encoder-decoder-based text architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
            "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> list[DummyInputGenerator]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config, **kwargs)
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
            self.task,
            self._normalized_config,
            **kwargs,
        )
        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2](
            self.task,
            self._normalized_config,
            encoder_sequence_length=dummy_text_input_generator.sequence_length,
            **kwargs,
        )
        return [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        """Fill input_or_outputs mapping with past_key_values dynamic axes for seq2seq."""
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + decoder_sequence_length"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch_size", 2: "encoder_sequence_length"}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch_size", 2: "encoder_sequence_length"}


class VisionOnnxConfig(OnnxConfig):
    """Handles vision architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)


class TextAndVisionOnnxConfig(OnnxConfig):
    """Handles multi-modal text and vision architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator, DummyBboxInputGenerator)


class AudioOnnxConfig(OnnxConfig):
    """Handles audio architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"input_values": {0: "batch_size", 1: "sequence_length"}}


class AudioToTextOnnxConfig(OnnxConfigWithPast):
    """Handles audio-to-text architectures like Whisper."""

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyAudioInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {
            "input_features": {0: "batch_size", 1: "feature_size", 2: "encoder_sequence_length"},
            "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        """Fill input_or_outputs mapping with past_key_values dynamic axes for audio-to-text."""
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + decoder_sequence_length"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch_size", 2: "encoder_sequence_length"}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch_size", 2: "encoder_sequence_length"}
