# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Type, Union, get_args

from pydantic import validator

from olive.common.config_utils import ConfigBase, ParamCategory, validate_config
from olive.common.user_module_loader import UserModuleLoader
from olive.data.config import DataConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec
from olive.model import CompositeOnnxModel, DistributedOnnxModel, OliveModel
from olive.passes.pass_config import (
    PassConfigBase,
    PassConfigParam,
    PassParamDefault,
    create_config_class,
    get_user_script_config,
)
from olive.resource_path import ResourcePath
from olive.strategy.search_parameter import (
    Categorical,
    Conditional,
    ConditionalDefault,
    SearchParameter,
    SpecialParamValue,
)
from olive.strategy.search_space import SearchSpace
from olive.strategy.utils import cyclic_search_space, order_search_parameters

logger = logging.getLogger(__name__)

# ruff: noqa: B027


class Pass(ABC):
    """Base class for pass configuration.

    Each pass should derive its own configuration class that contains all information it needs to execute.
    """

    registry: ClassVar[Dict[str, Type["Pass"]]] = {}
    # True if pass configuration requires user script for non-local host support
    _requires_user_script: bool = False
    # True if the pass processes a composite model at once. Otherwise, the components of the
    # composite model will be processed individually.
    _accepts_composite_model: bool = False

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the Pass."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls.registry[cls.__name__.lower()] = cls

    def __init__(
        self, accelerator_spec: AcceleratorSpec, config: Dict[str, Any], disable_search: Optional[bool] = False
    ):
        """Initialize the pass.

        :param config_class: the PassConfig class with the default value or default search values.
        :type config_class: Type[PassConfigBase]
        :param config: the configuration representing search space.
        :type config: Dict[str, Any]
        """
        assert accelerator_spec is not None, "Please specify the accelerator spec for the pass."
        assert config is not None, "Please specify the configuration for the pass."

        config_class, default_config = self.get_config_class(accelerator_spec, disable_search)

        self.accelerator_spec = accelerator_spec

        self._config_class = config_class
        self._config = config
        if self._requires_user_script:
            self._user_module_loader = UserModuleLoader(self._config["user_script"], self._config["script_dir"])

        self._fixed_params = {}
        self._search_space = {}
        for k, v in self._config.items():
            if isinstance(v, SearchParameter):
                self._search_space[k] = v
            else:
                self._fixed_params[k] = v

        # Params that are paths [(param_name, required)]
        self.path_params = []
        for param, param_config in default_config.items():
            if param_config.category in (ParamCategory.PATH, ParamCategory.DATA):
                self.path_params.append((param, param_config.required, param_config.category))

        self._initialized = False

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Whether the pass is accelerator agnostic. If True, the pass will be reused for all accelerators.

        The default value is True. The subclass could choose to override this method to return False by using the
        accelerator spec information.
        """
        return True

    @classmethod
    def generate_search_space(
        cls,
        accelerator_spec: AcceleratorSpec,
        config: Optional[Union[Dict[str, Any], PassConfigBase]] = None,
        disable_search: Optional[bool] = False,
    ) -> Tuple[Type[PassConfigBase], Dict[str, Any]]:
        """Generate search space for the pass."""
        assert accelerator_spec is not None, "Please specify the accelerator spec for the pass"

        # Get the config class with default value or default search value
        config_class, default_config = cls.get_config_class(accelerator_spec, disable_search)
        # Generate the search space by using both default value and default search value and user provided config
        config = validate_config(config, PassConfigBase, config_class)

        config = cls._resolve_config(config, default_config)
        return cls._init_fixed_and_search_params(config, default_config)

    @classmethod
    def get_config_class(cls, accelerator_spec: AcceleratorSpec, disable_search: Optional[bool] = False):
        """Get the configuration class for the pass."""
        default_config = cls.default_config(accelerator_spec)
        config_class = create_config_class(cls.__name__, default_config, disable_search, cls._validators())
        return config_class, default_config

    @classmethod
    def default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        """Get the default configuration for the pass."""
        config = {}
        if cls._requires_user_script:
            # add user script related parameters
            config.update(get_user_script_config())
        # add all other parameters
        config.update(cls._default_config(accelerator_spec))
        # validate that all parameters ending with data_config are of type DataConfig, Union[DataConfig, dict], ...
        # this requirement is on the pass developer but we can only check it here
        for param, param_config in config.items():
            if param.endswith("data_config"):
                param_type = param_config.type_
                assert param_type == DataConfig or DataConfig in get_args(
                    param_type
                ), f"{param} ending with data_config must be of type DataConfig."
        return config

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        """Pydantic validators for config params."""
        return {}

    @staticmethod
    @abstractmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        """Get the default configuration for the pass. Doesn't include user_script and script_dir.

        Example:
            return {
                # required parameter
                "param1": PassConfigParam(type_=int, required=True, description="param1 description"),
                # optional parameter with default value
                "param2": PassConfigParam(type_=int, default_value=1, description="param2 description"),
                # optional parameter with default value and searchable values
                "param3": PassConfigParam(
                    type_=int,
                    default_value=1,
                    searchable_values=Categorical([1, 2, 3]),
                    description="param3 description",
                ),
                # optional parameter with `category` set to `object`
                # the value of this parameter can be a string or a function that takes a string and returns the object,
                # say a class ObjectClass
                "param4": PassConfigParam(
                    type_=Union[str, Callable[[str], Pass]],
                    category=ParamCategory.OBJECT,
                    description="param4 description"
                ),
                # optional parameter with default_value that depends on another parameter value
                "param5": PassConfigParam(
                    type_=int,
                    default_value=ConditionalDefault(parents="param2", support={(1,): 2, (2,): 3}, default=4),
                    description="param5 description",
                ),
                # optional parameter with searchable_values that depends on other parameter values
                "param6": PassConfigParam(
                    type_=int,
                    default_value=1,
                    searchable_values=Conditional(
                        parents=("param2", "param3"),
                        # invalid if (param2, param3) not in [(1, 1), (1, 2)]
                        support={
                            (1, 1): Categorical([1, 2, 3]),
                            (1, 2): Categorical([4, 5, 6]),
                        },
                    ),
                    description="param6 description",
                ),
            }
        """
        raise NotImplementedError

    @classmethod
    def _resolve_defaults(cls, config: Dict[str, Any], default_config: Dict[str, PassConfigParam]) -> Dict[str, Any]:
        """Resolve default values."""
        for key, value in config.items():
            if value == PassParamDefault.DEFAULT_VALUE:
                config[key] = default_config[key].default_value
            elif value == PassParamDefault.SEARCHABLE_VALUES:
                v = default_config[key].searchable_values
                if v is None:
                    logger.warning(f"Parameter {key} does not have searchable values. Using default value instead.")
                    v = default_config[key].default_value
                config[key] = v
        return config

    @classmethod
    def _validate_user_script(
        cls, config: Dict[str, Any], user_module_loader: UserModuleLoader, default_config: Dict[str, PassConfigParam]
    ) -> Dict[str, Any]:
        """Validate callables in the config."""
        for key, value in config.items():
            if default_config[key].category == ParamCategory.OBJECT and isinstance(value, str):
                assert user_module_loader.user_script, f"'user_script' must be specified if a {key} is a string."
        # TODO(jambayk): once convention for user_script and script dir is finalized, let config class handle
        # currently, Olive cannot have other types of pytorch models (entire model, custom loader, etc) + hf_config
        # the resolution during serialization
        if config["user_script"] is not None:
            config["user_script"] = str(Path(config["user_script"]).resolve())
        if config["script_dir"] is not None:
            config["script_dir"] = str(Path(config["script_dir"]).resolve())
        return config

    @classmethod
    def _init_fixed_and_search_params(
        cls, config: Dict[str, Any], default_config: Dict[str, PassConfigParam]
    ) -> Tuple[Dict[str, Any], Dict[str, SearchParameter]]:
        """Get the fixed and search parameters from the config."""
        param_order = order_search_parameters(config)

        # fixed parameters
        fixed_params = {}
        search_space = {}
        for key in param_order:
            value = config[key]
            if isinstance(value, SearchParameter):
                # resolve conditional parameters
                # if categorical with single choice, use that choice directly
                value = cls._resolve_search_parameter(value, fixed_params)
            if value == SpecialParamValue.INVALID:
                # TODO(jambayk): better error message, e.g. what the parent values were, how it was invalid
                raise ValueError(
                    f"Invalid value for parameter '{key}'. Either the parameter or its parents are not fixed."
                )
            if isinstance(value, SearchParameter):
                search_space[key] = value
            else:
                if default_config[key].category == ParamCategory.PATH and value is not None:
                    if not isinstance(value, ResourcePath):
                        value = str(Path(value).resolve())
                fixed_params[key] = value
        assert not cyclic_search_space(search_space), "Search space is cyclic."
        # TODO(jambayk): better error message, e.g. which parameters are invalid, how they are invalid
        assert SearchSpace({"search_space": search_space}).size() > 0, "There are no valid points in the search space."

        return {**fixed_params, **search_space}

    @classmethod
    def _resolve_search_parameter(cls, param: SearchParameter, fixed_params: Dict[str, Any]) -> Any:
        """Resolve a search parameter."""
        if isinstance(param, Conditional):
            # if value is conditional and one/more parents are fixed, use the condition to get new value
            parent_values = {parent: fixed_params[parent] for parent in param.parents if parent in fixed_params}
            if len(parent_values) > 0:
                param = param.condition(parent_values)
            if isinstance(param, ConditionalDefault):
                # if there are still searchable parents, convert to conditional
                param = ConditionalDefault.conditional_default_to_conditional(param)
        if isinstance(param, Categorical) and len(param.get_support()) == 1:
            # if there is only one choice, use that choice
            param = param.get_support()[0]
        return param

    @classmethod
    def _resolve_config(
        cls,
        input_config: Union[Dict[str, Any], PassConfigBase],
        default_config: Dict[str, PassConfigParam],
    ) -> Dict[str, Any]:
        """Resolve config to PassConfigBase."""
        config = input_config.dict()
        config = cls._resolve_defaults(config, default_config)
        if cls._requires_user_script:
            user_module_loader = UserModuleLoader(config["user_script"], config["script_dir"])
            config = cls._validate_user_script(config, user_module_loader, default_config)
        return config

    def _initialize(self):
        """Initialize the pass. Pass specific initialization should be done here."""

    def search_space(self) -> Dict[str, SearchParameter]:
        """Get the search space for the pass."""
        return self._search_space

    def config_at_search_point(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """Get the configuration for the pass at a specific point in the search space."""
        assert set(point.keys()) == set(self._search_space.keys()), "Search point is not in the search space."
        config = self._fixed_params.copy()
        for key, value in point.items():
            config[key] = value
        return self._config_class(**config).dict()

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        """Validate the search point for the pass."""
        return True

    def filter_ignored_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out ignored parameters."""
        return {key: value for key, value in config.items() if value != SpecialParamValue.IGNORED}

    @abstractmethod
    def _run_for_config(
        self, model: OliveModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> OliveModel:
        """Run the pass on the model with the given configuration."""
        raise NotImplementedError

    def run(
        self, model: OliveModel, data_root: str, output_model_path: str, point: Optional[Dict[str, Any]] = None
    ) -> OliveModel:
        """Run the pass on the model at a specific point in the search space."""
        point = point or {}
        config = self.config_at_search_point(point)

        if not self._initialized:
            self._initialize()
            self._initialized = True

        # Optimization pass still works on individual graphs.
        if isinstance(model, DistributedOnnxModel):
            output_filepaths = []
            for rank in range(0, model.ranks):
                input_rank_model = model.load_model(rank)
                rank_output_path = Path(output_model_path).with_suffix("") / str(rank)
                output_rank_model = self._run_for_config(input_rank_model, data_root, config, rank_output_path)
                output_filepaths.append(output_rank_model.model_path)
            output_model = DistributedOnnxModel(output_filepaths, inference_settings=model.inference_settings)
        elif isinstance(model, CompositeOnnxModel) and not self._accepts_composite_model:
            components = []
            component_names = []
            for cidx, child in enumerate(model.get_model_components()):
                component_output_path = Path(output_model_path).with_suffix("") / str(cidx)
                output_model_component = self._run_for_config(child, data_root, config, str(component_output_path))
                output_model_component.model_attributes = (
                    output_model_component.model_attributes or child.model_attributes
                )
                components.append(output_model_component)
                component_names.append(model.get_model_component_name(cidx))
            output_model = CompositeOnnxModel(components, component_names)
        else:
            output_model = self._run_for_config(model, data_root, config, output_model_path)
        # assumption: the model attributes from passes, if any, are more important than
        # the input model attributes, we should not update/extend anymore outside of the pass run
        output_model.model_attributes = output_model.model_attributes or model.model_attributes
        return output_model

    def serialize_config(self, config: Dict[str, Any], check_object: bool = False) -> str:
        """Serialize the configuration."""
        return self._config_class(**config).to_json(check_object)

    def to_json(self, check_object: bool = False) -> Dict[str, Any]:
        """Convert the pass to json."""
        return {
            "type": self.__class__.__name__,
            "disable_search": True,
            "accelerator": self.accelerator_spec.to_json(),
            "config": self.serialize_config(self._config, check_object),
        }


# TODO(jambayk): rename. We are using FullPassConfig since PassConfigBase already refers to inner config
class FullPassConfig(ConfigBase):
    type: str  # noqa: A003
    disable_search: bool = False
    accelerator: Dict[str, str] = None
    config: Dict[str, Any] = None

    @validator("type")
    def validate_type(cls, v):
        if v.lower() not in Pass.registry:
            raise ValueError(f"Unknown pass type {v}")
        return v

    def create_pass(self):
        pass_cls = Pass.registry[self.type.lower()]
        accelerator_spec = AcceleratorSpec(**self.accelerator)
        return pass_cls(accelerator_spec, self.config, self.disable_search)


# TODO(myguo): deprecate or remove this method by explicitly specify the accelerator_spec in the arguments
# instead of using the default argument.
def create_pass_from_dict(
    pass_cls: Type[Pass], config: Dict[str, Any] = None, disable_search=False, accelerator_spec: AcceleratorSpec = None
) -> Pass:
    """Create a pass from a dictionary."""
    if accelerator_spec is None:
        accelerator_spec = DEFAULT_CPU_ACCELERATOR

    config = pass_cls.generate_search_space(accelerator_spec, config, disable_search)
    return pass_cls(accelerator_spec, config, disable_search)
