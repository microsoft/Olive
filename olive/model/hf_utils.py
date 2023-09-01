# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import transformers
from pydantic import Field, validator
from transformers import AutoConfig, AutoModel, AutoTokenizer

from olive.common.config_utils import ConfigBase, ConfigWithExtraArgs
from olive.model.hf_mappings import MODELS_TO_MAX_LENGTH_MAPPING, TASK_TO_FEATURE
from olive.model.model_config import IOConfig

logger = logging.getLogger(__name__)


class HFComponent(ConfigBase):
    name: str
    io_config: Union[IOConfig, str, Dict[str, Any]]
    component_func: Union[str, Callable]
    dummy_inputs_func: Union[str, Callable]


class HFModelLoadingArgs(ConfigWithExtraArgs):
    """
    Arguments to pass to the `from_pretrained` method of the model class.

    Refer to https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2074
    """

    torch_dtype: str = Field(
        None,
        description=(
            "torch dtype to load the model under. Refer to `torch_dtype` in the docstring of"
            " `transformers.PreTrainedModel.from_pretrained` for more details."
        ),
    )
    # str suffices for torch.dtype, otherwise serializer won't recognize it
    device_map: Union[int, str, Dict] = Field(
        None,
        description=(
            "A map that specifies where each submodule should go. Refer to `device_map` in the docstring of"
            " `transformers.PreTrainedModel.from_pretrained` for more details."
        ),
    )
    # A dictionary device identifier to maximum memory
    max_memory: Dict = Field(
        None,
        description=(
            "A dictionary that specifies the maximum memory that can be used by each device. Refer to `max_memory`"
            " in the docstring of `transformers.PreTrainedModel.from_pretrained` for more details."
        ),
    )
    # analog to transformers.utils.quantization_config.QuantizationMethod
    quantization_method: str = Field(
        None,
        description=(
            "Quantization method to use. Currently supported methods are ['bitsandbytes', 'gptq']. Must be provided if"
            " quantization_config is provided."
        ),
    )
    # A dictionary of configuration parameters for quantization
    quantization_config: Dict = Field(
        None,
        description=(
            "A dictionary of configuration parameters for quantization. Must be provided if quantization_config is"
            " provided. Please refer to `transformers.BitsAndBytesConfig` and `transformers.GPTQConfig` for more"
            " details for the supported parameters."
        ),
    )
    # other kwargs to pass during model loading
    extra_args: Dict = Field(
        None,
        description=(
            "Other kwargs to pass to the .from_pretrained method of the model class. Values can be provided directly to"
            " this field as a dict or as keyword arguments to the config. Please refer to the docstring of"
            " `transformers.PreTrainedModel.from_pretrained` for more details on the supported parameters. Eg."
            " {'use_safetensors': True}"
        ),
    )

    @validator("torch_dtype", pre=True)
    def validate_torch_dtype(cls, v):
        if isinstance(v, torch.dtype):
            v = str(v).replace("torch.", "")
        return v

    @validator("device_map", pre=True)
    def validate_device_map(cls, v):
        if isinstance(v, torch.device):
            v = cls.device_to_str(v)
        elif isinstance(v, dict):
            v = {k: cls.device_to_str(v) for k, v in v.items()}
        return v

    @validator("quantization_config", pre=True, always=True)
    def validate_quantization_config(cls, v, values):
        if "quantization_method" not in values:
            # to ensure we don't get a KeyError
            raise ValueError("Invalid quantization_config")
        if (values["quantization_method"] and not v) or (not values["quantization_method"] and v):
            raise ValueError("quantization_config and quantization_method must be provided together")
        if not v:
            return v

        try:
            return cls.dict_to_quantization_config(values["quantization_method"], v).to_dict()
        except ImportError:
            # we don't want to fail since the pass target might have the correct transformers version
            logger.warning(
                f"Could not import the config class for quantization method {values['quantization_method']}. Skipping "
                " validation"
            )
            return v

    def get_loading_args(self):
        loading_args = {}
        # copy args that can be directly copied
        direct_copy_args = ["device_map", "max_memory"]
        for arg in direct_copy_args:
            if getattr(self, arg):
                loading_args[arg] = deepcopy(getattr(self, arg))
        # convert torch dtype to torch.dtype or "auto"
        if self.torch_dtype:
            loading_args["torch_dtype"] = self.get_torch_dtype()
        # convert quantization_config to the config class
        quantization_config = self.get_quantization_config()
        if quantization_config:
            loading_args["quantization_config"] = quantization_config
        # add extra args
        if self.extra_args:
            loading_args.update(deepcopy(self.extra_args))
        return loading_args

    def get_torch_dtype(self):
        v = self.torch_dtype
        if isinstance(v, str) and v != "auto":
            # get rid of torch. prefix, this might have been added when serializing
            v = v.replace("torch.", "")
            try:
                return getattr(torch, v)
            except AttributeError as e:
                raise ValueError(f"Invalid torch dtype {v}") from e
        return v

    def get_quantization_config(self):
        if not self.quantization_method or not self.quantization_config:
            return None
        return self.dict_to_quantization_config(self.quantization_method, self.quantization_config)

    @staticmethod
    def device_to_str(device):
        if isinstance(device, torch.device):
            device = str(device)
        return device

    @staticmethod
    def dict_to_quantization_config(quantization_method, config_dict):
        method_to_class_name = {"bitsandbytes": "BitsAndBytesConfig", "gptq": "GPTQConfig"}
        method_to_min_version = {
            # bitsandbytes exists from 4.27.0, but 4bit quantization is only supported from 4.30.0
            "bitsandbytes": "4.30.0",
            "gptq": "4.32.0",
        }
        if quantization_method not in method_to_class_name:
            raise ValueError(
                f"Unsupported quantization method {quantization_method}. Supported methods are"
                f" {list(method_to_class_name.keys())}"
            )

        try:
            config_cls = getattr(transformers, method_to_class_name[quantization_method])
        except AttributeError as e:
            raise ImportError(
                f"Quantization method {quantization_method} is not supported in transformers version"
                f" {transformers.__version__}. Recommended transformers version is"
                f" {method_to_min_version[quantization_method]} or above"
            ) from e

        # return unused kwargs doesn't work in catching unused args in config_dict
        # they just call config_cls(**config_dict), extras get absorbed in **kwargs
        config = config_cls.from_dict(config_dict, return_unused_kwargs=False)
        # we will do a manual check to see if there are unused kwargs
        # this works since config_cls is created as a dataclass
        extras = set(config_dict.keys()) - set(config.__dict__.keys())
        if extras:
            logger.warning(f"Unused kwargs in quantization_config: {extras}. Ignoring them")
        return config


class HFConfig(ConfigBase):
    model_name: str = None
    task: str = None
    # feature is optional if task is specified and don't need past
    # else, provide feature such as "causal-lm-with-past"
    feature: str = None
    # TODO: remove model_class and only use task
    model_class: str = None
    components: List[HFComponent] = None
    dataset: Dict[str, Any] = None
    model_loading_args: HFModelLoadingArgs = None

    @validator("model_class", always=True)
    def task_or_model_class_required(cls, v, values):
        if values["model_name"]:
            if not v and not values.get("task", None):
                raise ValueError("Either task or model_class must be specified")
        return v

    def load_model(self, model_path: str = None):
        """Load model from model_path or model_name"""
        model_name_or_path = model_path or self.model_name
        loading_args = self.model_loading_args.get_loading_args() if self.model_loading_args else {}
        if self.task:
            model = load_huggingface_model_from_task(self.task, model_name_or_path, **loading_args)
        elif self.model_class:
            model = load_huggingface_model_from_model_class(self.model_class, model_name_or_path, **loading_args)
        return model

    def load_model_config(self, model_path: str = None):
        """Load model config from model_path or model_name"""
        model_name_or_path = model_path or self.model_name
        return get_hf_model_config(model_name_or_path)


def load_huggingface_model_from_task(task: str, name: str, **kwargs):
    """Load huggingface model from task and name"""
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
            logger.debug(f"Loaded model {model_class} with name_or_path {name}")
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
            raise AttributeError(f"{model_loader} is not found in transformers")
    elif not isinstance(model_loader, Callable):
        raise ValueError("model_loader must be a callable or a string defined in transformers")

    return model_loader.from_pretrained


def get_hf_model_config(model_name: str):
    """
    Get HF Config for the given model name
    """
    return AutoConfig.from_pretrained(model_name)


def load_huggingface_model_from_model_class(model_class: str, name: str, **kwargs):
    """
    Load huggingface model from model_loader and name
    """
    kwargs = kwargs or {}
    return huggingface_model_loader(model_class)(name, **kwargs)


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

    # if feature is not provided, try to get it from task
    # else use "default"
    feature = feature or TASK_TO_FEATURE.get(task, "default")

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
    model_config = get_onnx_config(model_name, task, feature)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_config.generate_dummy_inputs(tokenizer, framework="pt")


def get_model_max_length(model_name: str, fail_on_not_found=False) -> int:
    """
    Get max length of the model, extracted from the config
    """
    model_config = get_hf_model_config(model_name)
    model_type = model_config.model_type

    max_length = MODELS_TO_MAX_LENGTH_MAPPING.get(model_type, None)
    if isinstance(max_length, int):
        return max_length
    elif isinstance(max_length, str):
        return getattr(model_config, max_length)
    else:
        logger.debug(
            f"No max length mapping found in MODELS_TO_MAX_LENGTH_MAPPING for model type {model_type}, trying"
            " __default__"
        )
        default_max_length = MODELS_TO_MAX_LENGTH_MAPPING["__default__"]
        try:
            return getattr(model_config, default_max_length)
        except AttributeError:
            if fail_on_not_found:
                raise ValueError(f"Could not find max length for model type {model_type}")
            else:
                return None
