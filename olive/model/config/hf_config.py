# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
import transformers

from olive.common.config_utils import NestedConfig
from olive.common.pydantic_v1 import Field, validator
from olive.common.utils import exclude_keys, resolve_torch_dtype

logger = logging.getLogger(__name__)


class HfLoadKwargs(NestedConfig):
    """Arguments to pass to the `from_pretrained` method of the model class.

    Refer to https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
    """

    _nested_field_name = "extra_args"

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
            full_config = cls.dict_to_quantization_config(values["quantization_method"], v).to_dict()
            # in newer versions to_dict has extra keys quant_method, _load_in_4bit and _load_in_8bit
            # which are internal attributes and not part of init params
            for key in ["quant_method", "_load_in_4bit", "_load_in_8bit"]:
                full_config.pop(key, None)
            return full_config
        except ImportError:
            # we don't want to fail since the pass target might have the correct transformers version
            logger.warning(
                "Could not import the config class for quantization method %s. Skipping validation",
                values["quantization_method"],
            )
            return v

    def get_load_kwargs(self, exclude_load_keys: Optional[List[str]] = None) -> Dict[str, Any]:
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
        # exclude keys
        if exclude_load_keys:
            loading_args = exclude_keys(loading_args, exclude_load_keys)

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
