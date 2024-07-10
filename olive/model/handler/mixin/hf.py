# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from olive.model.utils.hf_utils import (
    get_hf_model_config,
    get_hf_model_dummy_input,
    get_hf_model_generation_config,
    get_hf_model_io_config,
    get_hf_model_tokenizer,
    save_hf_model_config,
    save_hf_model_tokenizer,
)

if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class HfMixin:
    """Provide the following Hugging Face model functionalities."""

    def get_load_kwargs(self) -> Dict[str, Any]:
        """Return all args from load_kwargs in a dict with types expected by `from_pretrained`."""
        return self.load_kwargs.get_load_kwargs() if self.load_kwargs else {}

    def get_hf_model_config(self) -> "PretrainedConfig":
        return get_hf_model_config(self.model_path, **self.get_load_kwargs())

    def get_hf_model_generation_config(self) -> "GenerationConfig":
        return get_hf_model_generation_config(self.model_path, **self.get_load_kwargs())

    def get_hf_model_tokenizer(self) -> Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]:
        # don't provide loading args for tokenizer directly since it tries to serialize all kwargs
        # TODO(anyone): only provide relevant kwargs, no use case for now to provide kwargs
        return get_hf_model_tokenizer(self.model_path)

    def save_metadata(self, output_dir: str, **kwargs) -> List[str]:
        """Save model metadata files to the output directory.

        :param output_dir: output directory to save metadata files
        :param kwargs: additional keyword arguments to pass to `save_pretrained` method
        :return: list of file paths
        """
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise ValueError("Expecting a directory as input.")

        saved_filepaths = []

        # save model config
        save_hf_model_config(self.get_hf_model_config(), output_dir, **kwargs)
        saved_filepaths.append(str(output_dir / "config.json"))

        # save model generation config
        # non-generative models won't have generation config
        generation_config = self.get_hf_model_generation_config()
        if generation_config:
            save_hf_model_config(generation_config, output_dir, **kwargs)
            saved_filepaths.append(str(output_dir / "generation_config.json"))

        # save tokenizer
        tokenizer_filepaths = save_hf_model_tokenizer(self.get_hf_model_tokenizer(), output_dir, **kwargs)
        saved_filepaths.extend([fp for fp in tokenizer_filepaths if Path(fp).exists()])

        return saved_filepaths

    def get_hf_io_config(self) -> Optional[Dict[str, Any]]:
        """Get Io config for the model."""
        return get_hf_model_io_config(self.model_path, self.task, **self.get_load_kwargs())

    def get_hf_dummy_inputs(self) -> Optional[Dict[str, Any]]:
        """Get dummy inputs for the model."""
        return get_hf_model_dummy_input(
            self.model_path,
            self.task,
            **self.get_load_kwargs(),
        )

    def get_hf_model_type(self) -> str:
        """Get model type for the model."""
        return self.get_hf_model_config().model_type
