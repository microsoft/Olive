# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

from pydantic import validator

from olive.common.auto_config import AutoConfigClass
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.user_module_loader import UserModuleLoader
from olive.model import OliveModel
from olive.passes.pass_config import (
    PassConfigBase,
    PassConfigParam,
    PassParamDefault,
    create_config_class,
    get_user_script_config,
)
from olive.strategy.search_parameter import Conditional, SearchParameter
from olive.strategy.utils import cyclic_search_space


class Pass(AutoConfigClass):
    """
    Base class for pass configuration.
    Each pass should derive its own configuration class that contains all information it needs to execute.
    """

    registry: Dict[str, "Pass"] = {}
    # True if pass configuration requires user script for non-local host support
    _requires_user_script: bool = False

    def __init__(self, config: Union[Dict[str, Any], PassConfigBase], default_to_search: Optional[bool] = False):
        """
        Initialize the pass.
        default_to_search: If True, use default search parameters, if any, for parameters that are not specified
        in the config. Only applies when the config is a dictionary.
        """
        self._config_class = self.get_config_class(default_to_search)
        self._config = validate_config(config, PassConfigBase, self._config_class)
        self._config = self._config.dict()
        self._config = self._resolve_defaults(self._config)
        if self._requires_user_script:
            self._user_module_loader = UserModuleLoader(self._config["user_script"], self._config["script_dir"])
            self._config = self._validate_user_script(self._config, self._user_module_loader)

        self._fixed_params, self._search_space = self._init_fixed_and_search_params(self._config)

        # Params that are paths [(param_name, required)]
        self.path_params = []
        for param, param_config in self.default_config().items():
            if param_config.is_path:
                self.path_params.append((param, param_config.required))

        self._initialized = False

    @classmethod
    def get_config_class(cls, default_to_search: Optional[bool] = False) -> Type[PassConfigBase]:
        """
        Get the configuration class for the pass.
        """
        return create_config_class(cls.__name__, cls.default_config(), default_to_search, cls._validators())

    @classmethod
    def default_config(cls) -> Dict[str, PassConfigParam]:
        """
        Get the default configuration for the pass.
        """
        config = {}
        if cls._requires_user_script:
            config.update(get_user_script_config())
        return {**config, **cls._default_config()}

    @staticmethod
    @abstractmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        """
        Get the default configuration for the pass. Doesn't include user_script and script_dir.
        """
        raise NotImplementedError()

    def _resolve_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve default values.
        """
        default_config = self.default_config()
        for key, value in config.items():
            if value == PassParamDefault.DEFAULT:
                config[key] = default_config[key].default
            elif value == PassParamDefault.DEFAULT_SEARCH:
                default_search = default_config[key].default_search
                assert default_search is not None, f"Parameter {key} does not have a default search."
                config[key] = default_search
        return config

    def _validate_user_script(self, config: Dict[str, Any], user_module_loader: UserModuleLoader) -> Dict[str, Any]:
        """
        Validate callables in the config.
        """
        default_config = self.default_config()
        for key, value in config.items():
            if default_config[key].is_object and isinstance(value, str):
                assert user_module_loader is not None, f"'user_script' must be specified if a {key} is a string."
        # TODO: once convention for user_script and script dir is finalized, let config class handle
        # the resolution during serialization
        if config["user_script"] is not None:
            config["user_script"] = str(Path(config["user_script"]).resolve())
        if config["script_dir"] is not None:
            config["script_dir"] = str(Path(config["script_dir"]).resolve())
        return config

    def _init_fixed_and_search_params(
        self, config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, SearchParameter]]:
        """
        Get the fixed and search parameters from the config.
        """
        default_config = self.default_config()

        # fixed parameters
        fixed_params = {}
        for key, value in config.items():
            if isinstance(value, SearchParameter):
                continue
            if default_config[key].is_path and value is not None:
                value = str(Path(value).resolve())
            fixed_params[key] = value

        # search parameters
        search_space = {}
        for key, value in config.items():
            if isinstance(value, SearchParameter):
                search_space[key] = self._resolve_search_parameter(value, fixed_params)
        assert not cyclic_search_space(search_space), "Search space is cyclic."

        return fixed_params, search_space

    def _resolve_search_parameter(self, param: SearchParameter, fixed_params: Dict[str, Any]) -> Any:
        """
        Resolve a search parameter.
        """
        if isinstance(param, Conditional):
            # if value is conditional and one/more parents are fixed, use the condition to get new value
            parent_values = {parent: fixed_params[parent] for parent in param.parents if parent in fixed_params}
            if len(parent_values) == 0:
                return param
            else:
                return param.condition(parent_values)
        else:
            return param

    def _initialize(self):
        """
        Initialize the pass. Pass specific initialization should be done here.
        """
        pass

    def search_space(self) -> Dict[str, SearchParameter]:
        """
        Get the search space for the pass.
        """
        return self._search_space

    def config_at_search_point(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the configuration for the pass at a specific point in the search space.
        """
        assert set(point.keys()) == set(self._search_space.keys()), "Search point is not in the search space."
        config = self._fixed_params.copy()
        for key, value in point.items():
            config[key] = value
        return self._config_class(**config).dict()

    def validate_search_point(self, search_point: Dict[str, Any]) -> bool:
        """
        Validate the search point for the pass.
        """
        return True

    @abstractmethod
    def _run_for_config(self, model: OliveModel, config: Dict[str, Any], output_model_path: str) -> OliveModel:
        """
        Run the pass on the model with the given configuration.
        """
        raise NotImplementedError()

    def run(self, model: OliveModel, output_model_path: str, point: Optional[Dict[str, Any]] = None) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        point = point or {}
        config = self.config_at_search_point(point)

        if not self._initialized:
            self._initialize()
            self._initialized = True

        return self._run_for_config(model, config, output_model_path)

    def serialize_config(self, config: Dict[str, Any], check_objects: bool = False) -> str:
        """
        Serialize the configuration.
        """
        return self._config_class(**config).to_json(check_objects)

    def to_json(self, check_objects: bool = False) -> Dict[str, Any]:
        """
        Convert the pass to json.
        """
        return {"type": self.__class__.__name__, "config": self.serialize_config(self._config, check_objects)}


# TODO rename. We are using FullPassConfig since PassConfigBase already refers to inner config
class FullPassConfig(ConfigBase):
    type: str
    default_to_search: bool = False
    config: PassConfigBase = None

    @validator("type")
    def validate_type(cls, v):
        if v.lower() not in Pass.registry:
            raise ValueError(f"Unknown pass type {v}")
        return v

    @validator("config", pre=True, always=True)
    def validate_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")
        if "default_to_search" not in values:
            raise ValueError("Invalid default_to_search")

        pass_type = values["type"].lower()
        default_to_search = values["default_to_search"]
        config_class = Pass.registry[pass_type].get_config_class(default_to_search)
        return validate_config(v, PassConfigBase, config_class)

    def create_pass(self):
        return Pass.registry[self.type.lower()](self.config, self.default_to_search)
