# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Dict, Optional, Tuple, Union

import transformers
from pydantic import Field, validator

from olive.common.config_utils import ConfigBase

logger = logging.getLogger(__name__)

LARGE_INTEGER = int(1e20)


class TokenizerConfig(ConfigBase):
    """Configuration parameters for HuggingFace tokenizers"""

    padding_side: str = Field("right", description="Padding side: right or left")
    padding: Union[bool, str, transformers.utils.PaddingStrategy] = Field(
        True,
        description=(
            "Activates and controls padding. Refer to"
            " https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.padding"  # noqa: E501
        ),
    )
    pad_to_multiple_of: int = Field(None, "Pad to multiple of this value. Only used if padding is activated.")
    truncation_side: str = Field("right", description="Truncation side: right or left")
    truncation: Union[bool, str, transformers.tokenization_utils_base.PaddingStrategy] = Field(
        True,
        description=(
            "Activates and controls truncation. Refer to"
            " https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.truncation"  # noqa: E501
        ),
    )
    add_special_tokens: bool = Field(
        True, description="Whether or not to encode the sequences with the special tokens relative to their model."
    )
    use_eos_token_as_pad_token: bool = Field(
        False,
        description=(
            "Use eos token as pad token, only valid if the tokenizer does not have a pad token won't mask the actual"
            " eos token, so the model can still generate eos tokens"
        ),
    )

    @validator("padding_side", "truncation_side")
    def validate_side(cls, v, field):
        if v not in ["right", "left"]:
            raise ValueError(f"{field.name} must be either right or left")
        return v

    @validator("padding", "truncation", pre=True, always=True)
    def validate_padding_truncation(cls, v, field):
        from transformers.tokenization_utils_base import TruncationStrategy
        from transformers.utils import PaddingStrategy

        enum_cls_map = {
            "padding": PaddingStrategy,
            "truncation": TruncationStrategy,
        }
        bool_map = {
            "padding": {
                True: PaddingStrategy.LONGEST,
                False: PaddingStrategy.DO_NOT_PAD,
            },
            "truncation": {
                True: TruncationStrategy.LONGEST_FIRST,
                False: TruncationStrategy.DO_NOT_TRUNCATE,
            },
        }

        field_name = field.name
        if isinstance(v, bool):
            v = bool_map[field_name][v]
        elif isinstance(v, str):
            v = enum_cls_map[field_name](v)
        return v

    def get_tokenizer(
        self, model_name_or_path: str, max_length: Optional[int] = None
    ) -> Tuple[transformers.PreTrainedTokenizer, Dict]:
        """Get tokenizer and kwargs to pass during tokenization"""
        from transformers import AutoTokenizer
        from transformers.tokenization_utils_base import TruncationStrategy
        from transformers.utils import PaddingStrategy

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=self.padding_side)

        # set pad token if needed
        if self.use_eos_token_as_pad_token:
            if tokenizer.pad_token:
                logger.warning(
                    "use_eos_token_as_pad_token is True, but tokenizer has a pad_token. "
                    "The pad_token will be used instead of the eos_token."
                )
            elif not tokenizer.eos_token:
                raise ValueError("use_eos_token_as_pad_token is True, but tokenizer does not have an eos_token")
            else:
                logger.debug("Using eos_token as pad_token")
                tokenizer.pad_token = tokenizer.eos_token

        # similar logic to PreTrainedTokenizerBase._get_padding_truncation_strategies
        # don't want to call the function since it's private and also falls back to default values if
        # pad_token doesn't exist

        # get the max length to use for padding/truncation
        model_max_length = tokenizer.model_max_length if tokenizer.model_max_length < LARGE_INTEGER else None
        relevant_max_length = max_length or model_max_length

        # check if max_length is available if needed
        if self.padding == PaddingStrategy.MAX_LENGTH and relevant_max_length is None:
            raise ValueError(
                "Padding is set to 'max_length' but no max_length is provided and the model has no predefined maximum"
                " length."
            )
        if self.truncation != TruncationStrategy.DO_NOT_TRUNCATE and relevant_max_length is None:
            raise ValueError(
                "Truncation is activated but no max_length is provided and the model has no predefined maximum length."
            )

        # check that pad token is defined if needed
        if self.padding != PaddingStrategy.DO_NOT_PAD and (tokenizer.pad_token is None or tokenizer.pad_token_id < 0):
            raise ValueError(
                "Padding is activated but tokenizer does not have a pad_token. Set use_eos_token_as_pad_token to"
                " True or add a pad_token to the tokenizer"
            )

        # check that relevant_max_length is multiple of pad_to_multiple_of if both padding and truncation are activated
        if (
            self.truncation != TruncationStrategy.DO_NOT_TRUNCATE
            and self.padding != PaddingStrategy.DO_NOT_PAD
            and self.pad_to_multiple_of is not None
            and relevant_max_length is not None
            and (relevant_max_length % self.pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but truncation length ({relevant_max_length}) is not a"
                f" multiple of pad_to_multiple_of ({self.pad_to_multiple_of})."
            )

        # tokenization kwargs
        kwargs = {
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": relevant_max_length,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "add_special_tokens": self.add_special_tokens,
        }
        return tokenizer, kwargs


def truncate_dataset(dataset, max_samples=None):
    """Truncate dataset to max_samples."""
    if max_samples is not None:
        num_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
    return dataset


def map_pre_process(dataset, pre_process_func, batched=True, remove_cols=True):
    """
    Apply pre-process function to dataset and set format to torch.

    :param dataset: Data to be pre-processed. Is expected to be of type datasets.Dataset.
    :param pre_process_func: Pre-process function to be applied to data. Takes a single example or a batch of examples
    as input.
    :param remove_cols: Whether to remove original columns from data after pre-processing.
    :param kwargs: Additional arguments to be passed to pre-process function.
    :return: Pre-processed data.
    """
    pre_processed_data = dataset.map(
        pre_process_func,
        batched=batched,
        remove_columns=dataset.column_names if remove_cols else None,
    )
    pre_processed_data.set_format("torch", output_all_columns=True)
    return pre_processed_data
