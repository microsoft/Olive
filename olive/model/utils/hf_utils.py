# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from functools import partial
from itertools import chain
from typing import Callable, Dict, Optional

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from olive.model.utils.hf_mappings import FEATURE_TO_PEFT_TASK_TYPE, MODELS_TO_MAX_LENGTH_MAPPING, TASK_TO_FEATURE

logger = logging.getLogger(__name__)


def load_hf_model_from_task(task: str, name: str, **kwargs):
    """Load huggingface model from task and name."""
    from transformers.pipelines import check_task

    task_results = check_task(task)
    assert isinstance(task_results, tuple)
    if len(task_results) == 2:
        targeted_task = task_results[0]
    elif len(task_results) == 3:
        targeted_task = task_results[1]
    else:
        raise ValueError("unsupported transformers version")

    model_class = {"pt": targeted_task["pt"]}
    class_tuple = ()
    class_tuple = class_tuple + model_class.get("pt", (AutoModel,))

    model = None
    for model_class in class_tuple:
        try:
            model = model_class.from_pretrained(name, **kwargs)
            logger.debug("Loaded model %s with name_or_path %s", model_class, name)
            return model
        except (OSError, ValueError):
            # the ValueError need to be caught since there will be multiple model_class for single task.
            # if the model_class is not the one for the task, it will raise ValueError and
            # next model_class will be tried.
            continue

    return model


def huggingface_model_loader(model_loader):
    if model_loader is None:
        model_loader = "AutoModel"
    if isinstance(model_loader, str):
        try:
            model_loader = getattr(transformers, model_loader)
        except AttributeError:
            raise AttributeError(f"{model_loader} is not found in transformers") from None
    elif not isinstance(model_loader, Callable):
        raise ValueError("model_loader must be a callable or a string defined in transformers")

    return model_loader.from_pretrained


def get_hf_model_config(model_name: str, **kwargs):
    """Get HF Config for the given model name."""
    return AutoConfig.from_pretrained(model_name, **kwargs)


def load_hf_model_from_model_class(model_class: str, name: str, **kwargs):
    """Load huggingface model from model_loader and name."""
    return huggingface_model_loader(model_class)(name, **kwargs)


# patched version of transformers.onnx.features.supported_features_mapping
# to support additional models in olive
def patched_supported_features_mapping(*supported_features: str, onnx_config_cls: str = None) -> Dict[str, Callable]:
    """Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.

    """
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    import olive.model.utils.hf_onnx_config as config_cls

    for attr_name in onnx_config_cls.split("."):
        config_cls = getattr(config_cls, attr_name)
    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)

    return mapping


def get_onnx_config(model_name: str, task: str, feature: Optional[str] = None, **kwargs):
    # pylint: disable=protected-access
    from transformers.onnx import FeaturesManager

    from olive.model.utils.hf_onnx_config import ADDITIONAL_MODEL_TYPES

    # patch FeaturesManager._SUPPORTED_MODEL_TYPE to support additional models in olive
    for model_type, feature_list in ADDITIONAL_MODEL_TYPES.items():
        if model_type in FeaturesManager._SUPPORTED_MODEL_TYPE:
            continue
        features, onnx_config_cls = feature_list
        FeaturesManager._SUPPORTED_MODEL_TYPE[model_type] = patched_supported_features_mapping(
            *features, onnx_config_cls=onnx_config_cls
        )

    # if feature is not provided, try to get it from task
    # else use "default"
    feature = feature or TASK_TO_FEATURE.get(task, "default")

    # don't want to load the model here since all we need is the config
    # model loading is expensive computationally and memory-wise for large models
    config = get_hf_model_config(model_name, **kwargs)
    # recreate the logic for FeaturesManager.check_supported_model_or_raise to get the model_onnx_config
    # https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L712
    model_type = config.model_type.replace("_", "-")
    model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
    if feature not in model_features:
        raise ValueError(
            f"{config.model_type} doesn't support feature {feature}. Supported values are: {model_features}"
        )
    return FeaturesManager.get_config(model_type, feature)(config)


def get_hf_model_io_config(model_name: str, task: str, feature: Optional[str] = None, **kwargs):
    model_config = get_onnx_config(model_name, task, feature, **kwargs)
    inputs = model_config.inputs
    outputs = model_config.outputs
    if not inputs or not outputs:
        # just log a warning and return None, since this is not a critical error
        # and following pass may not use the io_config, like OptimumConversion
        logger.warning("No inputs or outputs found from model %s", model_config)
        return None

    io_config = {}
    io_config["input_names"] = list(inputs.keys())
    io_config["output_names"] = list(outputs.keys())
    io_config["dynamic_axes"] = dict(chain(inputs.items(), outputs.items()))
    return io_config


def get_hf_model_dummy_input(model_name: str, task: str, feature: Optional[str] = None, **kwargs):
    model_config = get_onnx_config(model_name, task, feature, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    return model_config.generate_dummy_inputs(tokenizer, framework="pt")


def get_peft_task_type_from_task(task: str, fail_on_not_found=False) -> str:
    """Get peft task type from feature."""
    feature = TASK_TO_FEATURE.get(task, None)
    peft_task_type = FEATURE_TO_PEFT_TASK_TYPE.get(feature, None) if feature else None
    not_found_msg = f"There is no peft task type for task {task}"
    if peft_task_type is None and fail_on_not_found:
        raise ValueError(not_found_msg)
    elif peft_task_type is None:
        logger.warning(not_found_msg)
    return peft_task_type


def get_model_max_length(model_name: str, fail_on_not_found=False) -> int:
    """Get max length of the model, extracted from the config."""
    model_config = get_hf_model_config(model_name)
    model_type = model_config.model_type

    max_length = MODELS_TO_MAX_LENGTH_MAPPING.get(model_type, None)
    if isinstance(max_length, int):
        return max_length
    elif isinstance(max_length, str):
        return getattr(model_config, max_length)
    else:
        logger.debug(
            "No max length mapping found in MODELS_TO_MAX_LENGTH_MAPPING for model type %s, trying __default__",
            model_type,
        )
        default_max_length = MODELS_TO_MAX_LENGTH_MAPPING["__default__"]
        try:
            return getattr(model_config, default_max_length)
        except AttributeError:
            not_found_msg = f"Could not find max length for model type {model_type}"
            if fail_on_not_found:
                raise ValueError(not_found_msg) from None
            else:
                logger.warning(not_found_msg)
                return None
