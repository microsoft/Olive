# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from transformers import AutoConfig, AutoModel, AutoTokenizer, GenerationConfig

from olive.common.hf.mappings import TASK_TO_PEFT_TASK_TYPE
from olive.common.hf.mlflow import get_pretrained_name_or_path
from olive.common.utils import hardlink_copy_file

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def load_model_from_task(task: str, model_name_or_path: str, **kwargs) -> "PreTrainedModel":
    """Load huggingface model from task and model_name_or_path."""
    from transformers.pipelines import check_task

    task_results = check_task(task.replace("-with-past", ""))
    assert isinstance(task_results, tuple)
    if len(task_results) == 2:
        targeted_task = task_results[0]
    elif len(task_results) == 3:
        targeted_task = task_results[1]
    else:
        raise ValueError("unsupported transformers version")

    model_config = get_model_config(model_name_or_path, **kwargs)
    if getattr(model_config, "quantization_config", None):
        if not isinstance(model_config.quantization_config, dict):
            model_config.quantization_config = model_config.quantization_config.to_dict()

        if model_config.quantization_config.get("quant_method") == "gptq":
            config_use_exllama = model_config.quantization_config.get("use_exllama")
            # Force use_exllama to False for GPTQ quantized models unless it's provided as load kwargs
            # we mostly use quantized model for export and finetuning. Both don't support exllama
            kwargs_use_exllama = kwargs.pop("use_exllama", None) or False
            if config_use_exllama != kwargs_use_exllama:
                model_config.quantization_config["use_exllama"] = kwargs_use_exllama
                # provide config to the model load kwargs to override the saved config
                kwargs["config"] = model_config

    class_tuple = targeted_task["pt"] or (AutoModel,)
    model = None
    for i, model_class in enumerate(class_tuple):
        try:
            model = from_pretrained(model_class, model_name_or_path, "model", **kwargs)
            logger.debug("Loaded model %s with name_or_path %s", model_class, model_name_or_path)
            break
        except (OSError, ValueError) as e:
            if i == len(class_tuple) - 1:
                # len(class_tuple) == 1 covers most common tasks like text-generation, text-classification, etc
                # error could be device OOM, device_map: "auto" not supported, etc

                # len(class_tuple) > 1: not common - image-segmentation, conversational, etc
                # there is no easy way to get tracebacks for earlier failures, so just raise from last
                raise
            # the ValueError need to be caught since there will be multiple model_class for single task.
            # if the model_class is not the one for the task, it will raise ValueError and
            # next model_class will be tried.
            logger.info(
                "Failed to load model %s with name_or_path %s.\n kwargs: %s.\n Exception raised: %s",
                model_class,
                model_name_or_path,
                kwargs,
                e,
            )

    # this won't be None since class_tuple is never empty and we only reach here if model loaded successfully
    # satisfies linter too
    return model


def from_pretrained(cls, model_name_or_path: str, mlflow_dir: str, **kwargs):
    """Call cls.from_pretrained with hf checkpoint or mlflow model.

    If the model_name_or_path is an MLFlow model, the corresponding subdirectory is used.
    """
    return cls.from_pretrained(get_pretrained_name_or_path(model_name_or_path, mlflow_dir), **kwargs)


def get_model_config(model_name_or_path: str, **kwargs) -> "PretrainedConfig":
    """Get HF Config for the given model_name_or_path."""
    model_config = from_pretrained(AutoConfig, model_name_or_path, "config", **kwargs)

    # add quantization config
    quantization_config = kwargs.get("quantization_config")
    if not quantization_config:
        return model_config

    if hasattr(model_config, "quantization_config") and model_config.quantization_config:
        logger.warning(
            "Model config already has quantization_config but new quantization_config is provided as load kwargs."
            " Ignoring the new quantization_config. This quantization config might be different from the one used by"
            " the model."
        )
    else:
        model_config.quantization_config = quantization_config
    return model_config


def save_model_config(config: Union["PretrainedConfig", "GenerationConfig"], output_dir: str, **kwargs):
    """Save input HF Config to output directory."""
    config.save_pretrained(output_dir, **kwargs)


def save_module_files(
    config: "PretrainedConfig", model_name_or_path: str, output_dir: str, **kwargs
) -> Tuple["PretrainedConfig", List[str]]:
    """Save module files for the given model_name_or_path.

    Returns updated config and list of saved module files.
    """
    # check if auto_map is present in the config
    auto_map = getattr(config, "auto_map", None)
    if not auto_map:
        return config, []

    # save the module files
    module_files = set()
    for key, value in config.auto_map.items():
        try:
            class_reference, module_file = get_module_file(value, model_name_or_path, **kwargs)

            # save the module file
            save_path = Path(output_dir) / Path(module_file).name
            hardlink_copy_file(module_file, save_path)
            module_files.add(str(save_path))

            # -- is attached with hf-id, don't save it in config, from_pretrained will try to load it
            # from the model hub
            config.auto_map[key] = class_reference
        except Exception as e:
            logger.warning(
                "Failed to save module file for %s: %s. Loading config with `trust_remote_code=True` will fail!",
                value,
                e,
            )
    # set _auto_class to None to avoid saving the module file again and causing SameFileError error
    config._auto_class = None  # pylint: disable=protected-access

    return config, list(module_files)


# https://github.com/huggingface/transformers/blob/9d6c0641c4a3c2c5ecf4d49d7609edd5b745d9bc/src/transformers/dynamic_module_utils.py#L493
# we could get the file from the loaded class but that requires us to force trust_remote_code=True
def get_module_file(
    class_reference: str,
    model_name_or_path: str,
    code_revision: Optional[str] = None,
    revision: Optional[str] = None,
    **kwargs,
) -> Tuple[str, str]:
    """Get module file for the given class_reference.

    Returns class_reference and module file path.
    """
    from transformers.dynamic_module_utils import HF_MODULES_CACHE, get_cached_module_file

    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = model_name_or_path

    # class reference is of form configuration_phi3.Phi3Config
    module_name = class_reference.split(".")[0]

    # get code revision
    if code_revision is None and model_name_or_path == repo_id:
        code_revision = revision

    # get the module file
    module_file = Path(HF_MODULES_CACHE) / get_cached_module_file(
        repo_id, f"{module_name}.py", revision=code_revision, **kwargs
    )

    return class_reference, str(module_file)


def get_generation_config(model_name_or_path: str, **kwargs) -> Optional["GenerationConfig"]:
    """Get HF model's generation config for the given model_name_or_path. If not found, return None."""
    try:
        return from_pretrained(GenerationConfig, model_name_or_path, "model", **kwargs)
    except OSError:
        return None


def get_tokenizer(model_name_or_path: str, **kwargs) -> Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]:
    """Get HF model's tokenizer."""
    tokenizer = from_pretrained(AutoTokenizer, model_name_or_path, "tokenizer", **kwargs)
    if getattr(tokenizer, "pad_token", None) is None:
        logger.debug("Setting pad_token to eos_token for tokenizer %s", model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_tokenizer(
    tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"], output_dir: str, **kwargs
) -> Tuple[str]:
    """Save input tokenizer to output directory."""
    return tokenizer.save_pretrained(output_dir, **kwargs)


def get_peft_task_type_from_task(task: str, fail_on_not_found=False) -> str:
    """Get peft task type from task."""
    peft_task_type = TASK_TO_PEFT_TASK_TYPE.get(task.replace("-with-past", ""), None)
    not_found_msg = f"There is no peft task type for task {task}"
    if peft_task_type is None and fail_on_not_found:
        raise ValueError(not_found_msg)
    elif peft_task_type is None:
        logger.warning(not_found_msg)
    return peft_task_type
