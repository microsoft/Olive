# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Type, Union

from olive.common.config_utils import ConfigBase, ConfigParam, ParamCategory, validate_object, validate_resource_path
from olive.common.pydantic_v1 import Field, create_model, validator
from olive.strategy.search_parameter import SearchParameter, json_to_search_parameter

logger = logging.getLogger(__name__)


class PassParamDefault(str, Enum):
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
    searchable_values: default searchable values for the parameter. This value is used if search is enabled.
        Must be a Categorical or Conditional SearchParameter.
    """

    searchable_values: SearchParameter = None
    allowed_values: Set = None

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


# TODO(jambayk): set types for user_script and script_dir once we decide on a convention
def get_user_script_config(
    required: Optional[bool] = False, allow_path: Optional[bool] = False
) -> Dict[str, PassConfigParam]:
    type_ = str
    if allow_path:
        type_ = Union[Path, str]

    user_script_config = {
        "script_dir": PassConfigParam(
            type_=type_,
            required=required,
            category=ParamCategory.PATH,
            description="Directory containing user script dependencies.",
        ),
        "user_script": PassConfigParam(
            type_=type_,
            required=required,
            category=ParamCategory.PATH,
            description=(
                "Path to user script. The values for other parameters which were assigned function or object names will"
                " be imported from this script."
            ),
        ),
    }
    return user_script_config  # noqa: RET504


class PassConfigBase(ConfigBase):
    @validator("*", pre=True)
    def _validate_default_str(cls, v, field):
        try:
            v = PassParamDefault(v)
        finally:
            if field.required and isinstance(v, PassParamDefault):
                raise ValueError(f"{field.name} is required and cannot be set to {v.value}")
            return v  # noqa: B012 # pylint: disable=lost-exception, return-in-finally

    @validator("*", pre=True)
    def _validate_search_parameter(cls, v):
        if isinstance(v, dict) and v.get("olive_parameter_type") == "SearchParameter":
            return json_to_search_parameter(v)
        return v


def validate_allowed_values(v, values, field):
    if v == PassParamDefault.DEFAULT_VALUE or v == PassParamDefault.SEARCHABLE_VALUES or isinstance(v, SearchParameter):
        # skip validation for PassParamDefault and SearchParameter
        # will be validated later when the actual value is set
        return v

    allowed_values = field.field_info.extra.get("allowed_values")
    if allowed_values is None:
        logger.warning(f"validate allowed values is called for {field.name} but allowed values is not set")
        return v
    if v not in allowed_values:
        raise ValueError(f"{v} is not in supported values {allowed_values}")
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
            validators[f"validate_object_{param}"] = validator(param, allow_reuse=True)(validate_object)
        if param == "data_dir":
            validators[f"validate_resource_path_{param}"] = validator(param, allow_reuse=True)(validate_resource_path)
        if param_config.allowed_values is not None:
            validators[f"validate_allowed_values_{param}"] = validator(param, allow_reuse=True)(validate_allowed_values)

        type_ = param_config.type_
        if param_config.required:
            config[param] = (type_, Field(..., allowed_values=param_config.allowed_values))
            continue

        type_ = Optional[Union[type_, SearchParameter, PassParamDefault]]
        if not disable_search and param_config.searchable_values is not None:
            config[param] = (type_, Field(param_config.searchable_values, allowed_values=param_config.allowed_values))
        else:
            config[param] = (type_, Field(param_config.default_value, allowed_values=param_config.allowed_values))

    return create_model(f"{pass_type}Config", **config, __base__=PassConfigBase, __validators__=validators)
