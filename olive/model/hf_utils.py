# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import FieldValidationInfo, field_validator

from olive.common.config_utils import ConfigBase
from olive.model.model_config import IOConfig

logger = logging.getLogger(__name__)


class HFComponent(ConfigBase):
    name: str
    io_config: Union[IOConfig, str, Dict[str, Any]]
    component_func: Union[str, Callable]
    dummy_inputs_func: Union[str, Callable]


class HFConfig(ConfigBase):
    model_name: Optional[str] = None
    task: Optional[str] = None
    feature: Optional[str] = None
    # TODO: remove model_class and only use task
    model_class: Optional[str] = None
    components: Optional[List[HFComponent]] = None
    config: Optional[Dict[str, Any]] = None
    dataset: Optional[Dict[str, Any]] = None

    @field_validator("model_class")
    def task_or_model_class_required(cls, v, info: FieldValidationInfo):
        if info.data["model_name"]:
            if not v and not info.data.get("task", None):
                raise ValueError("Either task or model_class must be specified")
        return v


def load_huggingface_model_from_task(task: str, name: str):
    """Load huggingface model from task and name"""
    from transformers import AutoModel
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
            model = model_class.from_pretrained(name)
            logger.debug(f"Loaded model {model_class} with name_or_path {name}")
            return model
        except (OSError, ValueError):
            # the ValueError need to be caught since there will be multiple model_class for single task.
            # if the model_class is not the one for the task, it will raise ValueError and
            # next model_class will be tried.
            continue

    return model


def huggingface_model_loader(model_loader):
    import transformers

    if model_loader is None:
        model_loader = "AutoModel"
    if isinstance(model_loader, str):
        try:
            model_loader = getattr(transformers, model_loader)
        except AttributeError:
            raise AttributeError(f"{model_loader} is not found in transformers")
    elif not isinstance(model_loader, Callable):
        raise ValueError("model_loader must be a callable or a string defined in transformers")

    return model_loader.from_pretrained


def get_hf_model_config(model_name: str):
    """
    Get HF Config for the given model name
    """
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name)


def load_huggingface_model_from_model_class(model_class: str, name: str):
    """
    Load huggingface model from model_loader and name
    """
    return huggingface_model_loader(model_class)(name)


# patched version of transforrmers.onnx.features.supported_features_mapping
# to support additional models in olive
def patched_supported_features_mapping(*supported_features: str, onnx_config_cls: str = None) -> Dict[str, Callable]:
    """
    Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.
    """
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    import olive.model.hf_onnx_config as config_cls

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


def get_onnx_config(model_name: str, task: str, feature: Optional[str] = None):
    from transformers.onnx import FeaturesManager

    from olive.model.hf_onnx_config import ADDITIONAL_MODEL_TYPES

    # patch FeaturesManager._SUPPORTED_MODEL_TYPE to support additional models in olive
    for model_type in ADDITIONAL_MODEL_TYPES:
        if model_type in FeaturesManager._SUPPORTED_MODEL_TYPE:
            continue
        features, onnx_config_cls = ADDITIONAL_MODEL_TYPES[model_type]
        FeaturesManager._SUPPORTED_MODEL_TYPE[model_type] = patched_supported_features_mapping(
            *features, onnx_config_cls=onnx_config_cls
        )

    # mapping from task to feature
    task_to_feature = {
        "automatic-speech-recognition": "speech2seq-lm",
        "fill-mask": "masked-lm",
        "image-classification": "image-classification",
        "image-segmentation": "image-segmentation",
        "image-to-text": "vision2seq-lm",
        "multiple-choice": "multiple-choice",
        "ner": "token-classification",
        "object-detection": "object-detection",
        "question-answering": "question-answering",
        "sentiment-analysis": "sequence-classification",
        "summarization": "seq2seq-lm",
        "text2text-generation": "seq2seq-lm",
        "text-classification": "sequence-classification",
        "text-generation": "causal-lm",
        "token-classification": "token-classification",
        "translation": "seq2seq-lm",
    }
    # if feature is not provided, try to get it from task
    # else use "default"
    feature = feature or task_to_feature.get(task, "default")

    # don't want to load the model here since all we need is the config
    # model loading is expensive computationally and memory-wise for large models
    config = get_hf_model_config(model_name)
    # recreate the logic for FeaturesManager.check_supported_model_or_raise to get the model_onnx_config
    # https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L712
    model_type = config.model_type.replace("_", "-")
    model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
    if feature not in model_features:
        raise ValueError(
            f"{config.model_type} doesn't support feature {feature}. Supported values are: {model_features}"
        )
    return FeaturesManager.get_config(model_type, feature)(config)


def get_hf_model_io_config(model_name: str, task: str, feature: Optional[str] = None):
    model_config = get_onnx_config(model_name, task, feature)
    inputs = model_config.inputs
    outputs = model_config.outputs
    io_config = {}
    io_config["input_names"] = list(inputs.keys())
    io_config["output_names"] = list(outputs.keys())
    io_config["dynamic_axes"] = dict(chain(inputs.items(), outputs.items()))
    return io_config


def get_hf_model_dummy_input(model_name: str, task: str, feature: Optional[str] = None):
    from transformers import AutoTokenizer

    model_config = get_onnx_config(model_name, task, feature)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_config.generate_dummy_inputs(tokenizer, framework="pt")
