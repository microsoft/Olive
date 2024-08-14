# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from olive.common.hf.model_io import get_model_dummy_input, get_model_io_config
from olive.common.hf.utils import (
    get_generation_config,
    get_model_config,
    get_tokenizer,
    save_model_config,
    save_module_files,
    save_tokenizer,
)

if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class HfMixin:
    """Provide the following Hugging Face model functionalities."""

    def get_load_kwargs(self, exclude_load_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return all args from load_kwargs in a dict with types expected by `from_pretrained`.

        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :return: dict of load_kwargs
        """
        return self.load_kwargs.get_load_kwargs(exclude_load_keys) if self.load_kwargs else {}

    def get_hf_model_config(self, exclude_load_keys: Optional[List[str]] = None) -> "PretrainedConfig":
        """Get model config for the model.

        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :return: model config
        """
        return get_model_config(self.model_path, **self.get_load_kwargs(exclude_load_keys))

    def get_hf_generation_config(self, exclude_load_keys: Optional[List[str]] = None) -> Optional["GenerationConfig"]:
        """Get generation config for the model if it exists.

        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :return: generation config or None
        """
        return get_generation_config(self.model_path, **self.get_load_kwargs(exclude_load_keys))

    def get_hf_tokenizer(self) -> Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]:
        """Get tokenizer for the model."""
        # don't provide loading args for tokenizer directly since it tries to serialize all kwargs
        # TODO(anyone): only provide relevant kwargs, no use case for now to provide kwargs
        return get_tokenizer(self.model_path)

    def save_metadata(self, output_dir: str, exclude_load_keys: Optional[List[str]] = None, **kwargs) -> List[str]:
        """Save model metadata files to the output directory.

        :param output_dir: output directory to save metadata files
        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :param kwargs: additional keyword arguments to pass to `save_pretrained` method
        :return: list of file paths
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        elif not output_dir.is_dir():
            raise ValueError("Expecting a directory as input.")

        saved_filepaths = []

        # save config and module files
        config = self.get_hf_model_config(exclude_load_keys=exclude_load_keys)
        if getattr(config, "auto_map", None):
            # needs model_name_or_path to find module files
            # conditional since model_name_or_path might trigger preprocessing for some mlflow models
            config, module_files = save_module_files(
                config,
                self.model_name_or_path,
                str(output_dir),
                **self.get_load_kwargs(exclude_load_keys=exclude_load_keys),
            )
            saved_filepaths.extend(module_files)
        save_model_config(config, output_dir, **kwargs)
        saved_filepaths.append(str(output_dir / "config.json"))

        # save model generation config
        # non-generative models won't have generation config
        generation_config = self.get_hf_generation_config(exclude_load_keys=exclude_load_keys)
        if generation_config:
            save_model_config(generation_config, output_dir, **kwargs)
            saved_filepaths.append(str(output_dir / "generation_config.json"))

        # save tokenizer
        tokenizer_filepaths = save_tokenizer(self.get_hf_tokenizer(), output_dir, **kwargs)
        saved_filepaths.extend([fp for fp in tokenizer_filepaths if Path(fp).exists()])

        return saved_filepaths

    def get_hf_io_config(self) -> Optional[Dict[str, Any]]:
        """Get Io config for the model."""
        return get_model_io_config(self.model_path, self.task, self.load_model(), **self.get_load_kwargs())

    def get_hf_dummy_inputs(self) -> Optional[Dict[str, Any]]:
        """Get dummy inputs for the model."""
        return get_model_dummy_input(
            self.model_path,
            self.task,
            **self.get_load_kwargs(),
        )

    def get_hf_model_type(self) -> str:
        """Get model type for the model."""
        return self.get_hf_model_config().model_type
