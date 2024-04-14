# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import torch
import transformers

from olive.common.config_utils import ConfigBase, ConfigWithExtraArgs
from olive.common.pydantic_v1 import Field, validator
from olive.common.utils import resolve_torch_dtype
from olive.model.config.io_config import IoConfig

logger = logging.getLogger(__name__)


class HfComponent(ConfigBase):
    """Used for Hf models which has multiple components, such as whisper.

    For example, in the Whisper model example, the component looks like:
        {
            "name": "encoder_decoder_init",
            "io_config": "get_encdec_io_config",
            "component_func": "get_encoder_decoder_init",
            "dummy_inputs_func": "encoder_decoder_init_dummy_inputs"
        }
    """

    name: str
    io_config: Union[IoConfig, Dict[str, Any], str, Callable]
    component_func: Union[str, Callable] = None
    dummy_inputs_func: Union[str, Callable]


class HfFromPretrainedArgs(ConfigWithExtraArgs):
    """Arguments to pass to the `from_pretrained` method of the model class.

    Refer to https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
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
    # whether to trust remote code
    trust_remote_code: bool = Field(
        None,
        description=(
            "Whether to trust remote code. Refer to `trust_remote_code` in the docstring of"
            " `transformers.PreTrainedModel.from_pretrained` for more details."
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
                "Could not import the config class for quantization method %s. Skipping validation",
                values["quantization_method"],
            )
            return v

    def get_loading_args(self) -> Dict[str, Any]:
        """Return all args in a dict with types expected by `from_pretrained`."""
        loading_args = {}
        # copy args that can be directly copied
        direct_copy_args = ("device_map", "max_memory")
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
        if self.trust_remote_code:
            loading_args["trust_remote_code"] = self.trust_remote_code
        # add extra args
        if self.extra_args:
            loading_args.update(deepcopy(self.extra_args))
        return loading_args

    def get_torch_dtype(self):
        """Return the torch dtype to load the model under. It is either a torch.dtype or 'auto'."""
        v = self.torch_dtype
        if isinstance(v, str) and v != "auto":
            v = resolve_torch_dtype(v)
        return v

    def get_quantization_config(self):
        """Return the quantization config to use. It is either None or a config class."""
        if not self.quantization_method or not self.quantization_config:
            return None
        return self.dict_to_quantization_config(self.quantization_method, self.quantization_config)

    @staticmethod
    def device_to_str(device) -> str:
        if isinstance(device, torch.device):
            device = str(device)
        return device

    @staticmethod
    def dict_to_quantization_config(quantization_method: str, config_dict: Dict[str, Any]):
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
        extras = set()
        for k in config_dict:
            if not hasattr(config, k):
                extras.add(k)
        if extras:
            logger.warning("Unused kwargs in quantization_config: %s. Ignoring them", extras)
        return config


class HfConfig(ConfigBase):
    """The config for HuggingFace models.

    For example, the config for the Whisper model looks like:
        "model_class": "WhisperForConditionalGeneration",
        "model_name": "openai/whisper-tiny.en",
        "components": [
            {
                "name": "encoder_decoder_init",
                "io_config": "get_encdec_io_config",
                "component_func": "get_encoder_decoder_init",
                "dummy_inputs_func": "encoder_decoder_init_dummy_inputs"
            },
            {
                "name": "decoder",
                "io_config": "get_dec_io_config",
                "component_func": "get_decoder",
                "dummy_inputs_func": "decoder_dummy_inputs"
            }
        ]
    """

    model_name: str = None
    task: str = None
    # feature is optional if task is specified and don't need past
    # else, provide feature such as "causal-lm-with-past"
    feature: str = None
    # TODO(xiaoyu): remove model_class and only use task
    model_class: str = None
    components: List[HfComponent] = None
    dataset: Dict[str, Any] = None
    from_pretrained_args: HfFromPretrainedArgs = None

    @validator("model_class", always=True)
    def task_or_model_class_required(cls, v, values):
        if values["model_name"] and not v and not values.get("task", None):
            raise ValueError("Either task or model_class must be specified")
        return v

    def get_loading_args_from_pretrained(self) -> Dict[str, Any]:
        """Return all args from from_pretrained_args in a dict with types expected by `from_pretrained`."""
        return self.from_pretrained_args.get_loading_args() if self.from_pretrained_args else {}


def get_model_type_from_hf_config(hf_config: HfConfig) -> str:
    from olive.model.utils.hf_utils import get_hf_model_config

    return get_hf_model_config(hf_config.model_name, **hf_config.get_loading_args_from_pretrained()).model_type
