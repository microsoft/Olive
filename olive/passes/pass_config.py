# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Set, Type, Union

from olive.common.config_utils import ConfigBase, ConfigParam, ParamCategory, validate_object
from olive.common.pydantic_v1 import Field, create_model, validator
from olive.common.utils import StrEnumBase
from olive.hardware.accelerator import Device
from olive.hardware.constants import DEVICE_TO_EXECUTION_PROVIDERS
from olive.resource_path import validate_resource_path
from olive.strategy.search_parameter import SearchParameter, SpecialParamValue, json_to_search_parameter


class PassParamDefault(StrEnumBase):
    """Default values for passes."""

    DEFAULT_VALUE = "DEFAULT_VALUE"
    SEARCHABLE_VALUES = "SEARCHABLE_VALUES"


class PassConfigParam(ConfigParam):
    """Dataclass for pass configuration parameters.

    Parameters
    ----------
    type_ : type of the parameter
    required : whether the parameter is required
    category : category of the parameter. it could be
        * object: whether the parameter is an object/function. If so, this parameter accepts the object or a string with
          the name of the object/function in the user script. The type must include str.
        * path : whether the parameter is a path. If so, this file/folder will be uploaded to the host system.
        * data: whether the parameter is a data path, which will be used to do path normalization based on the data root
    description : description of the parameter
    default_value: default value for the parameter. This value is used if search is disabled or there are no searchable
        values. Must be the same type as the parameter or a ConditionalDefault SearchParameter.
    search_defaults: default search defaults for the search parameter. This value is used if search is enabled and user
        hasn't provided any specific input to search on. Must be a Categorical or Conditional SearchParameter.

    """

    search_defaults: SearchParameter = None

    def __repr__(self):
        repr_list = []
        booleans = ["required"]
        for k, v in self.__dict__.items():
            if k in booleans:
                if v:
                    repr_list.append(f"{k}={v}")
            elif v is not None:
                repr_list.append(f"{k}={v}")
        return f"({', '.join(repr_list)})"


def get_user_script_data_config(
    required: Optional[bool] = False, allow_path: Optional[bool] = True
) -> Dict[str, PassConfigParam]:
    type_ = str
    if allow_path:
        type_ = Union[Path, str]

    user_script_config = {
        "user_script": PassConfigParam(
            type_=type_,
            required=required,
            category=ParamCategory.PATH,
            description=(
                "Path to user script. The values for other parameters which were assigned "
                "function or object names will be imported from this script."
            ),
        ),
        "script_dir": PassConfigParam(
            type_=type_,
            required=False,
            category=ParamCategory.PATH,
            description="Directory containing user script dependencies.",
        ),
    }
    return user_script_config  # noqa: RET504


DEFAULT_SET = set(PassParamDefault)


class PassConfigBase(ConfigBase):

    @validator("*", pre=True)
    def _validate_default_str(cls, v, field):
        if not isinstance(v, (str, PassParamDefault)) or v not in DEFAULT_SET:
            return v

        if field.required:
            raise ValueError(f"{field.name} is required and cannot be set to {v}")

        return PassParamDefault(v)

    @validator("*", pre=True)
    def _validate_search_parameter(cls, v):
        if isinstance(v, dict) and v.get("olive_parameter_type") == "SearchParameter":
            return json_to_search_parameter(v)
        return v


def create_config_class(
    pass_type: str,
    default_config: Dict[str, PassConfigParam],
    disable_search: Optional[bool] = False,
    validators: Dict[str, Callable] = None,
) -> Type[PassConfigBase]:
    """Create a Pydantic model class from a configuration dictionary."""
    config = {}
    validators = validators.copy() if validators else {}
    for param, param_config in default_config.items():
        if param_config.category == ParamCategory.OBJECT:
            validators[f"validate_{param}"] = validator(param, allow_reuse=True)(validate_object)
        if param == "data_dir":
            validators[f"validate_{param}"] = validator(param, allow_reuse=True)(validate_resource_path)

        type_ = param_config.type_
        if param_config.required:
            config[param] = (type_, ...)
            continue

        # Value can be one of
        # 1. Instance of type_ if search is disabled or search_defaults is None
        # 2. Search Parameter if search is enabled and search_defaults is not None
        # 3. PassParamDefault if value is set to "DEFAULT_VALUE" or "SEARCHABLE_VALUES"
        # 4. SpecialParamValue.IGNORED if the param is ignored for a specific search point. This is used to ignore
        #    parameters that are only used conditional on other parameters. Such as static quantization parameters
        #    that are only used if the quantization mode is static.
        type_ = Optional[Union[type_, SearchParameter, PassParamDefault, SpecialParamValue]]
        if not disable_search and param_config.search_defaults is not None:
            config[param] = (type_, param_config.search_defaults)
        else:
            config[param] = (type_, param_config.default_value)

    return create_model(f"{pass_type}Config", **config, __base__=PassConfigBase, __validators__=validators)


class PassModuleConfig(ConfigBase):
    class Precision(StrEnumBase):
        INT4 = "int4"
        INT8 = "int8"
        INT16 = "int16"
        INT32 = "int32"
        UINT4 = "uint4"
        UINT8 = "uint8"
        UINT16 = "uint16"
        UINT32 = "uint32"
        FP4 = "fp4"
        FP8 = "fp8"
        FP16 = "fp16"
        FP32 = "fp32"
        NF4 = "nf4"

    ACCELERATORS: ClassVar[Set[str]] = {v.value for v in Device}
    PRECISIONS: ClassVar[Set[str]] = {v.value for v in Precision}
    EXECUTION_PROVIDERS: ClassVar[Set[str]] = {
        provider for provider_list in DEVICE_TO_EXECUTION_PROVIDERS.values() for provider in provider_list
    }

    module_path: str
    supported_providers: Set[str] = Field(default_factory=set)
    supported_accelerators: Set[str] = Field(default_factory=set)
    supported_precisions: Set[str] = Field(default_factory=set)
    module_dependencies: List[str] = Field(default_factory=list)
    extra_dependencies: List[str] = Field(default_factory=list)

    @validator("module_path", pre=True)
    def validate_module_path(cls, v):
        if not v:
            raise ValueError("module_path cannot be empty or None")
        return v

    @validator("supported_providers", pre=True)
    def validate_supported_providers(cls, v, values):
        v = v or []
        if v == ["*"]:
            v = PassModuleConfig.EXECUTION_PROVIDERS
        return v

    @validator("supported_providers", pre=True, each_item=True)
    def validate_supported_provider(cls, v, values):
        if v not in PassModuleConfig.EXECUTION_PROVIDERS:
            raise ValueError(f"Invalid provider: {v}")
        return v

    @validator("supported_accelerators", pre=True)
    def validate_supported_accelerators(cls, v, values):
        v = v or []
        if v == ["*"]:
            v = PassModuleConfig.ACCELERATORS
        return v

    @validator("supported_accelerators", pre=True, each_item=True)
    def validate_supported_accelerator(cls, v, values):
        if v not in PassModuleConfig.ACCELERATORS:
            raise ValueError(f"Invalid accelerator: {v}")
        return v

    @validator("supported_precisions", pre=True)
    def validate_supported_precisions(cls, v, values):
        v = v or []
        if v == ["*"]:
            v = PassModuleConfig.PRECISIONS
        return v

    @validator("supported_precisions", pre=True, each_item=True)
    def validate_supported_precision(cls, v, values):
        if v not in PassModuleConfig.PRECISIONS:
            raise ValueError(f"Invalid precision: {v}")
        return v
