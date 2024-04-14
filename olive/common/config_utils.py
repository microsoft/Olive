# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import json
import logging
from enum import Enum
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from olive.common.pydantic_v1 import BaseModel, Field, create_model, root_validator, validator
from olive.common.utils import hash_function, hash_object

logger = logging.getLogger(__name__)


def serialize_function(function: Union[FunctionType, MethodType]) -> dict:
    """Serialize a function into a dictionary."""
    return {
        "olive_parameter_type": "Function",
        "name": function.__name__,
        "signature": str(inspect.signature(function)),
        "sourcecode_hash": hash_function(function),
    }


def serialize_object(obj: Any) -> dict:
    """Serialize an object into a dictionary."""
    return {
        "olive_parameter_type": "Object",
        "type": type(obj).__name__,
        "hash": hash_object(obj),
    }


def _expanded_default(custom_default: Callable[[Any], Any], obj: Any) -> Any:
    if custom_default is not None:
        try:
            return custom_default(obj)
        except TypeError:
            pass
    if isinstance(obj, (FunctionType, MethodType)):
        return serialize_function(obj)
    if isinstance(obj, Path):
        return str(obj.resolve())
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return serialize_object(obj)


def config_json_dumps(obj: Any, default: Callable[[Any], Any] = None, **kwargs) -> str:
    """Serialize a Python object into a JSON string. Also serializes functions and objects."""
    default = partial(_expanded_default, default)
    return json.dumps(obj, default=default, **kwargs)


def _expanded_object_hook(custom_object_hook: Callable[[dict], Any], obj: dict) -> Any:
    if obj.get("olive_parameter_type") in ["Function", "Object"]:
        param_type = obj.get("type", obj.get("olive_parameter_type"))
        raise ValueError(
            f"Cannot load a {param_type} from JSON. Replace {param_type} with user_script and name string."
        )
    if custom_object_hook is None:
        return obj
    return custom_object_hook(obj)


def config_json_loads(s: Union[str, bytes, bytearray], *, object_hook: Callable[[dict], Any] = None, **kwargs) -> Any:
    """Deserialize a JSON string into a Python object."""
    object_hook = partial(_expanded_object_hook, object_hook)
    return json.loads(s, object_hook=object_hook, **kwargs)


def serialize_to_json(obj: Any, check_object: bool = False) -> dict:
    """Serialize a Python object into a JSON dict. Also serializes functions and objects."""
    if isinstance(obj, BaseModel):
        raw_json = obj.json()
    else:
        raw_json = config_json_dumps(obj)
    if check_object:
        try:
            config_json_loads(raw_json)
        except ValueError as e:
            e = str(e)
            if "user_script" in e:
                e = e.replace("Cannot load", "Cannot serialize")
                e = e.replace("from JSON", "to JSON")
            raise ValueError(e) from None
    return json.loads(raw_json)


class ConfigBase(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_loads = config_json_loads
        json_dumps = config_json_dumps
        json_encoders = {Path: lambda x: str(x.resolve())}  # noqa: RUF012

    def to_json(self, check_object: bool = False) -> dict:
        return serialize_to_json(self, check_object)

    @classmethod
    def from_json(cls, json_dict: dict) -> "ConfigBase":
        return cls.parse_raw(json.dumps(json_dict))


class ConfigListBase(ConfigBase):
    __root__: List[Any]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__)


class ConfigDictBase(ConfigBase):
    __root__: Dict[str, Any]

    def __iter__(self):
        return iter(self.__root__)

    def keys(self):
        return self.__root__.keys()

    def values(self):
        return self.__root__.values()

    def items(self):
        return self.__root__.items()

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__) if self.__root__ else 0


class ConfigWithExtraArgs(ConfigBase):
    """Config class that automatically gathers all values.

    The values are not defined in the class fields and inserted into a dict Field called `extra_args`.
    """

    extra_args: Dict = Field(
        None,
        description=(
            "Dictionary of extra arguments that are not defined in the class fields. Values can be provided in two"
            " ways: 1. As a dict value to `extra_args` key. 2. As keyword arguments to the class constructor. Any"
            " values provided as keyword arguments will be added to the `extra_args` dict. `extra_args` values take"
            " precedence over keyword arguments if the same key is provided in both."
        ),
    )

    @root_validator(pre=True)
    def gather_extra_args(cls, values):
        other_fields = set()
        for field in cls.__fields__.values():
            for name in (field.name, field.alias):
                if name != "extra_args":
                    other_fields.add(name)

        extra_args = values.pop("extra_args", {}) or {}
        # ensure that extra_args does not contain any field names
        for name in list(extra_args):  # need a copy of the keys since we are mutating the dict
            if name in other_fields:
                logger.warning(
                    "'%s' provided to 'extra_args' is already defined in the class fields. Please provide the"
                    " value directly to the field. Ignoring.",
                    name,
                )
                del extra_args[name]
        # put any values provided as keyword arguments into extra_args
        for name in list(values):  # need a copy of the keys since we are mutating the dict
            if name in other_fields:
                continue
            if name in extra_args:
                # extra_args takes precedence over keyword arguments
                logger.warning("kwarg '%s' is already defined in 'extra_args'. Ignoring.", name)
            else:
                extra_args[name] = values.pop(name)
        if extra_args:
            values["extra_args"] = extra_args
        return values


class ParamCategory(str, Enum):
    NONE = "none"
    OBJECT = "object"
    PATH = "path"
    DATA = "data"

    def __str__(self) -> str:
        return self.value


class ConfigParam(ConfigBase):
    """Dataclass for pass configuration parameters."""

    type_: Any
    required: bool = False
    default_value: Any = None
    category: ParamCategory = ParamCategory.NONE
    description: str = None

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


# validator for enum params
def validate_enum(enum_class: type, value: str):
    try:
        value = enum_class(value)
    except ValueError:
        raise ValueError(f"Invalid value '{value}'. Valid values are {[e.value for e in enum_class]}") from None
    return value


# validator for object params. This ensures user_script is not None if value v is string
def validate_object(v, values, field):
    if "user_script" not in values:
        raise ValueError("Invalid user_script")
    if isinstance(v, str) and values["user_script"] is None:
        raise ValueError(f"user_script must be provided if {field.name} is a name string")
    return v


def validate_resource_path(v, values, field):
    from olive.resource_path import create_resource_path

    try:
        v = create_resource_path(v)
    except ValueError as e:
        raise ValueError(f"Invalid resource path '{v}': {e}") from None
    return v


def create_config_class(
    class_name: str,
    default_config: Dict[str, ConfigParam],
    base: type = ConfigBase,
    validators: Dict[str, Callable] = None,
) -> Type[ConfigBase]:
    """Create a Pydantic model class from a configuration dictionary."""
    config = {}
    validators = validators.copy() if validators else {}
    for param, param_config in default_config.items():
        if param == "data_dir":
            validator_name = f"validate_{param}_resource_path"
            validators[validator_name] = validator(param, allow_reuse=True)(validate_resource_path)
        # automatically add validator for object params
        if param_config.category == ParamCategory.OBJECT:
            validator_name = f"validate_{param}_object"
            count = 0
            while validator_name in validators:
                validator_name = f"{validator_name}_{count}"
                count += 1
            validators[validator_name] = validator(param, allow_reuse=True)(validate_object)

        type_ = param_config.type_
        if param_config.required:
            config[param] = (type_, ...)
            continue

        config[param] = (Optional[type_], param_config.default_value)

    return create_model(class_name, **config, __base__=base, __validators__=validators)


T = TypeVar("T", bound=ConfigBase)


def validate_config(
    config: Union[Dict[str, Any], ConfigBase, None],
    instance_class: Type[T],
    warn_unused_keys: bool = True,
) -> T:
    """Validate a config dictionary or object against a base class and instance class.

    instance class is a subclass of base class.
    """
    config = config or {}

    if isinstance(config, dict):
        user_keys = set(config.keys())
        config = instance_class(**config)
        config_keys = set(config.dict().keys())
        unused_keys = user_keys - config_keys
        if unused_keys and warn_unused_keys:
            logger.warning("Keys %s are not part of %s. Ignoring them.", unused_keys, instance_class.__name__)
    # for dynamically created class by Pydantic create_model, the classes are different even if the class names are same
    elif (
        isinstance(config, ConfigBase)
        and config.__class__.__module__ == instance_class.__module__
        and config.__class__.__name__ == instance_class.__name__
    ):
        pass
    else:
        raise ValueError(
            f"Invalid config class. Expected {instance_class.__name__} but got {config.__class__.__name__}"
        )
    return config
