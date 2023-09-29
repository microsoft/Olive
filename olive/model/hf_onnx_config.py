# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from typing import Any, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType
from transformers.onnx import OnnxConfigWithPast

# dictionary of model types and their (supported features list, config class name)
# the supported features list is the list of features that are supported by the model type
# similar to the tasks supported by the model type. Refer to `task_to_feature` under `get_onnx_config` in
# `hf_utils.py` for a mapping from task to feature.
ADDITIONAL_MODEL_TYPES = {
    "gpt-neox": (
        [
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
            "token-classification",
        ],
        "TextDecoderOnnxConfig",
    ),
    "llama": (
        [
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "sequence-classification",
        ],
        "TextDecoderOnnxConfig",
    ),
    "opt": (
        [
            "default",
            "default-with-past",
            "causal-lm",
            "causal-lm-with-past",
            "question-answering",
            "sequence-classification",
        ],
        "TextDecoderOnnxConfig",
    ),
}


class TextDecoderOnnxConfig(OnnxConfigWithPast):
    # in OnnxConfigWithPast.fill_with_past_key_values_
    # there is a bug in the name for the present sequence length dimension
    # it should be `past_sequence` instead of `past_sequence + sequence`
    def fill_with_past_key_values_(
        self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str, inverted_values_shape: bool = False
    ):
        """Fill the input_or_outputs mapping with past_key_values dynamic axes considering.

        Args:
            inputs_or_outputs: The mapping to fill.
            direction: either "inputs" or "outputs", it specifies whether input_or_outputs is the input mapping or the
                output mapping, this is important for axes naming.
            inverted_values_shape:
                If `True`, store values on dynamic axis 1, else on axis 2.

        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        sequence_length_name = "past_sequence" if direction == "inputs" else "past_sequence + sequence"
        for i in range(self.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch", 2: sequence_length_name}
            if inverted_values_shape:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 1: sequence_length_name}
            else:
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 2: sequence_length_name}

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            # there seems to be a bug in the size of the past_key_values dim 2
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        input_order = ["input_ids"]
        if self.use_past:
            input_order += ["past_key_values"]
        input_order += ["attention_mask"]

        return OrderedDict({k: common_inputs[k] for k in input_order})

    @property
    def num_layers(self) -> int:
        return self._config.num_hidden_layers
