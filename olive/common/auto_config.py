# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Type, Union

from olive.common.config_utils import ConfigBase, ConfigParam, create_config_class, validate_config


class AutoConfigClass(ABC):
    """Base class for creating other classes with easily extensible subclassing.

    Registry
    The class maintains a registry of all concrete subclasses.
    To refresh the registry for a child base class, e.g., SearchAlgorithm, just set registry = {}

    Dynamically created config class
    All classes are instantiated by passing a config dictionary or config class (BaseModel) instance.
    Sub-class developer just needs to implement the static method _default_config
    E.g.,
        @classmethod
        def _default_config(cls):
            return {
                "str_param": ConfigParam(type_=str, required=True),
                "func_param" ConfigParam(type_=Union[str, Callable], category=ParamCategory.OBJECT)
            }
    The class dynamically creates its config class through the class method `get_config_class`.
    This config class has validates for types and also automatically validates object/func params
    to ensure `script_dir` is present if the param value is string.

    Additional validators
    Sub-class developer can also add additional validators by implementing the static method _validators
    E.g.,
        from olive.common.pydantic_v1 import validator

        def validate_func_param(v, values):
            ...
            return v

        @classmethod
        def _validators(cls):
            return {"validate_func_param": validator("func_param", allow_reuse=True)(validate_func_param)}
    """

    registry: ClassVar[Dict[str, Type]] = {}
    name: str = None
    _config_base: Type[ConfigBase] = ConfigBase

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the metric."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        name = cls.name if cls.name is not None else cls.__name__.lower()
        cls.registry[name] = cls

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]]) -> None:
        self.config_class = self.get_config_class()
        self.config = validate_config(config, self.config_class)

    @classmethod
    @abstractmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        """Get the default configuration for the class."""
        raise NotImplementedError

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        """Get ydantic validators for config params."""
        return {}

    @classmethod
    def default_config(cls):
        """Get the default configuration."""
        assert not inspect.isabstract(cls), "Cannot get default config for abstract class"
        return cls._default_config()

    @classmethod
    def get_config_class(cls) -> Type[ConfigBase]:
        """Get the configuration class."""
        assert not inspect.isabstract(cls), "Cannot get config class for abstract class"
        return create_config_class(f"{cls.__name__}Config", cls.default_config(), cls._config_base, cls._validators())
