# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

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

    def get_load_kwargs(self, exclude_load_keys: Optional[list[str]] = None) -> dict[str, Any]:
        """Return all args from load_kwargs in a dict with types expected by `from_pretrained`.

        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :return: dict of load_kwargs
        """
        return self.load_kwargs.get_load_kwargs(exclude_load_keys) if self.load_kwargs else {}

    def get_hf_model_config(self, exclude_load_keys: Optional[list[str]] = None) -> "PretrainedConfig":
        """Get model config for the model.

        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :return: model config
        """
        return get_model_config(
            self.model_path,
            test_model_config=getattr(self, "test_model_config", None),
            **self.get_load_kwargs(exclude_load_keys),
        )

    def get_hf_generation_config(self, exclude_load_keys: Optional[list[str]] = None) -> Optional["GenerationConfig"]:
        """Get generation config for the model if it exists.

        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :return: generation config or None
        """
        # Generation config loading should not receive model-loading-only kwargs such as
        # dtype, device placement, or quantization settings.
        generation_config_exclude_keys = {"torch_dtype", "dtype", "device_map", "max_memory", "quantization_config"}
        if exclude_load_keys:
            generation_config_exclude_keys.update(exclude_load_keys)
        return get_generation_config(self.model_path, **self.get_load_kwargs(list(generation_config_exclude_keys)))

    def get_hf_tokenizer(self) -> Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]:
        """Get tokenizer for the model."""
        # don't provide loading args for tokenizer directly since it tries to serialize all kwargs
        # TODO(anyone): only provide relevant kwargs, no use case for now to provide kwargs
        return get_tokenizer(self.model_path)

    def save_metadata(self, output_dir: str, exclude_load_keys: Optional[list[str]] = None, **kwargs) -> list[str]:
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
        # load already saved config, could be saved by loaded_model.save_pretrained
        # don't want to override it with the current model config
        config_file_path = output_dir / "config.json"
        config = (
            get_model_config(output_dir, trust_remote_code=self.get_load_kwargs().get("trust_remote"))
            if config_file_path.exists()
            else self.get_hf_model_config(exclude_load_keys=exclude_load_keys)
        )
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
        saved_filepaths.append(str(config_file_path))

        # save model generation config, skip if it already exists
        # non-generative models won't have generation config
        generation_config_file_path = output_dir / "generation_config.json"
        if not generation_config_file_path.exists() and (
            generation_config := self.get_hf_generation_config(exclude_load_keys=exclude_load_keys)
        ):
            save_model_config(generation_config, output_dir, **kwargs)
            saved_filepaths.append(str(generation_config_file_path))

        # save tokenizer, skip if it already exists
        if not (output_dir / "tokenizer_config.json").exists():
            # there is no tokenizer in the output_dir, save the tokenizer
            tokenizer_filepaths = save_tokenizer(self.get_hf_tokenizer(), output_dir, **kwargs)
            saved_filepaths.extend([fp for fp in tokenizer_filepaths if Path(fp).exists()])

        # save processor / image processor; per-file, don't overwrite anything that already exists
        # (see ``_copy_missing_files``). This writes preprocessor_config.json (and any image
        # processor files) so downstream tools that load from this output_dir (e.g. mobius's
        # AutoProcessor.from_pretrained) get the model's real preprocessing config instead of
        # silently falling back to defaults. Only applicable to multimodal models (e.g. VL
        # checkpoints); text-only models have no processor and are already covered by the
        # tokenizer save above.
        #
        # Note: unlike the tokenizer save above, this is not gated on a single sentinel file
        # (e.g. "does preprocessor_config.json already exist?") -- a processor can emit several
        # files (preprocessor_config.json, chat_template.json, ...), and an earlier step may have
        # saved only some of them. Always calling ``_save_processor`` lets ``_copy_missing_files``
        # fill in whichever files are still missing, file by file.
        saved_filepaths.extend(self._save_processor(output_dir, exclude_load_keys=exclude_load_keys, **kwargs))

        logger.debug("Save metadata files to %s: %s", output_dir, saved_filepaths)

        return saved_filepaths

    def _save_processor(self, output_dir: Path, exclude_load_keys: Optional[list[str]] = None, **kwargs) -> list[str]:
        """Save the model's processor files (preprocessor_config.json, ...) to output_dir.

        Never overwrites a file that already exists in ``output_dir``:
        ``ProcessorMixin.save_pretrained`` also re-saves the processor's tokenizer, which would
        clobber a tokenizer that an earlier step intentionally customized and saved. The
        processor is therefore saved to a temporary directory first and only its new files are
        copied over.

        :param output_dir: output directory to save the processor files in
        :param exclude_load_keys: list of keys to exclude from load_kwargs
        :param kwargs: additional keyword arguments to pass to `save_pretrained` method
        :return: list of file paths that were written
        """
        from transformers import AutoProcessor
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, **self.get_load_kwargs(exclude_load_keys=exclude_load_keys)
            )
        except Exception as e:  # pylint: disable=broad-except
            # unexpected: loading failed for a reason other than "this model has no processor"
            # (network / auth / incompatible config). Surface it -- VL models genuinely need
            # preprocessor_config.json downstream.
            logger.warning("Failed to load processor for %r, no processor files saved: %s", self.model_name_or_path, e)
            return []

        if isinstance(processor, PreTrainedTokenizerBase):
            # expected for text-only models: AutoProcessor falls back to returning the
            # tokenizer, which the tokenizer save above already handled.
            logger.debug("No processor for %r (AutoProcessor returned a tokenizer).", self.model_name_or_path)
            return []

        try:
            with tempfile.TemporaryDirectory(prefix="olive_processor_") as temp_dir:
                processor.save_pretrained(temp_dir, **kwargs)
                return self._copy_missing_files(Path(temp_dir), output_dir)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to save processor files for %r: %s", self.model_name_or_path, e)
            return []

    @staticmethod
    def _copy_missing_files(src_dir: Path, output_dir: Path) -> list[str]:
        """Copy files from src_dir into output_dir, keeping any file that already exists there."""
        copied_filepaths = []
        for src_path in sorted(src_dir.rglob("*")):
            if not src_path.is_file():
                continue
            dst_path = output_dir / src_path.relative_to(src_dir)
            if dst_path.exists():
                logger.debug("Keeping existing %s instead of overwriting it.", dst_path)
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            copied_filepaths.append(str(dst_path))
        return copied_filepaths

    def get_hf_io_config(self) -> Optional[dict[str, Any]]:
        """Get Io config for the model."""
        return get_model_io_config(
            self.model_path,
            self.task,
            self.load_model(),
            test_model_config=getattr(self, "test_model_config", None),
            **self.get_load_kwargs(),
        )

    def get_hf_dummy_inputs(self) -> Optional[dict[str, Any]]:
        """Get dummy inputs for the model."""
        return get_model_dummy_input(
            self.model_path,
            self.task,
            model=self.load_model(),
            test_model_config=getattr(self, "test_model_config", None),
            **self.get_load_kwargs(),
        )

    def get_hf_model_type(self) -> str:
        """Get model type for the model."""
        return self.get_hf_model_config().model_type
