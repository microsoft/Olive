# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import json
import logging
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel, create_model, validator

from olive.common.utils import hash_function, hash_object

logger = logging.getLogger(__name__)


def serialize_function(function: Union[FunctionType, MethodType]) -> dict:
    """
    Serialize a function into a dictionary.
    """
    return {
        "olive_parameter_type": "Function",
        "name": function.__name__,
        "signature": str(inspect.signature(function)),
        "sourcecode_hash": hash_function(function),
    }


def serialize_object(obj: Any) -> dict:
    """
    Serialize an object into a dictionary.
    """
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
    if isinstance(obj, FunctionType) or isinstance(obj, MethodType):
        return serialize_function(obj)
    if isinstance(obj, Path):
        return str(obj.resolve())
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return serialize_object(obj)


def config_json_dumps(obj: Any, default: Callable[[Any], Any] = None, **kwargs) -> str:
    """
    Serialize a Python object into a JSON string. Also serializes functions and objects.
    """
    default = partial(_expanded_default, default)
    return json.dumps(obj, default=default, **kwargs)


def _expanded_object_hook(custom_object_hook: Callable[[dict], Any], obj: dict) -> Any:
    if obj.get("olive_parameter_type") in ["Function", "Object"]:
        type = obj.get("type", obj.get("olive_parameter_type"))
        raise ValueError(f"Cannot load a {type} from JSON. Replace {type} with user_script and name string.")
    if custom_object_hook is None:
        return obj
    return custom_object_hook(obj)


def config_json_loads(s: Union[str, bytes, bytearray], *, object_hook: Callable[[dict], Any] = None, **kwargs) -> Any:
    """
    Deserialize a JSON string into a Python object.
    """
    object_hook = partial(_expanded_object_hook, object_hook)
    return json.loads(s, object_hook=object_hook, **kwargs)


def serialize_to_json(obj: Any, check_objects: bool = False) -> dict:
    """
    Serialize a Python object into a JSON dict. Also serializes functions and objects.
    """
    if isinstance(obj, BaseModel):
        raw_json = obj.json()
    else:
        raw_json = config_json_dumps(obj)
    if check_objects:
        try:
            config_json_loads(raw_json)
        except ValueError as e:
            e = str(e)
            if "user_script" in e:
                e = e.replace("Cannot load", "Cannot serialize")
                e = e.replace("from JSON", "to JSON")
            raise ValueError(e)
    return json.loads(raw_json)


class ConfigBase(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_loads = config_json_loads
        json_dumps = config_json_dumps
        json_encoders = {Path: lambda x: str(x.resolve())}

    def to_json(self, check_objects: bool = False) -> dict:
        return serialize_to_json(self, check_objects)

    @classmethod
    def from_json(cls, json_dict: dict) -> "ConfigBase":
        return cls.parse_raw(json.dumps(json_dict))


class ConfigParam(ConfigBase):
    """
    Dataclass for pass configuration parameters.
    """

    type_: Any
    required: bool = False
    default_value: Any = None
    is_object: bool = False
    description: str = None

    def __repr__(self):
        repr_list = []
        booleans = ["required", "is_object"]
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
        raise ValueError(f"Invalid value '{value}'. Valid values are {[e.value for e in enum_class]}")
    return value


# validator for object params. This ensures user_script is not None if value v is string
def validate_object(v, values, field):
    if "user_script" not in values:
        raise ValueError("Invalid user_script")
    if isinstance(v, str) and values["user_script"] is None:
        raise ValueError(f"user_script must be provided if {field.name} is a name string")
    return v


def create_config_class(
    class_name: str,
    default_config: Dict[str, ConfigParam],
    base: type = ConfigBase,
    validators: Dict[str, Callable] = None,
):
    """
    Create a Pydantic model class from a configuration dictionary.
    """
    config = {}
    validators = validators.copy() if validators else {}
    for param, param_config in default_config.items():
        # automatically add validator for object params
        if param_config.is_object:
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


def validate_config(
    config: Union[Dict[str, Any], ConfigBase, None],
    base_class: ConfigBase,
    instance_class: Optional[ConfigBase] = None,
    warn_unused_keys: bool = True,
):
    """
    Validate a config dictionary or object against a base class and instance class.
    instance class is a subclass of base class.
    """
    config = config or {}

    if instance_class is None:
        instance_class = base_class

    if isinstance(config, dict):
        user_keys = set(config.keys())
        config = instance_class(**config)
        config_keys = set(config.dict().keys())
        unused_keys = user_keys - config_keys
        if unused_keys and warn_unused_keys:
            logger.warning(f"Keys {unused_keys} are not part of {instance_class.__name__}. Ignoring them.")
    elif isinstance(config, base_class) and config.__class__.__name__ == instance_class.__name__:
        pass
    else:
        raise ValueError(
            f"Invalid config class. Expected {instance_class.__name__} but got {config.__class__.__name__}"
        )
    return config
