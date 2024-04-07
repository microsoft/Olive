# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Type, Union, get_args

from olive.common.config_utils import ConfigBase, ParamCategory, validate_config
from olive.common.user_module_loader import UserModuleLoader
from olive.data.config import DataConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec
from olive.model import CompositeModelHandler, DistributedOnnxModelHandler, OliveModelHandler
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

    # Flag indicate whether the pass need to be run in target instead of host
    run_on_target: bool = False

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the Pass."""
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls.registry[cls.__name__.lower()] = cls

    def __init__(
        self,
        accelerator_spec: AcceleratorSpec,
        config: Dict[str, Any],
        disable_search: Optional[bool] = False,
        host_device=None,
    ):
        """Initialize the pass.

        :param accelerator_spec: the accelerator spec for the pass.
        :type accelerator_spec: AcceleratorSpec
        :param config: the configuration representing search space.
        :type config: Dict[str, Any]
        :param disable_search: whether to disable search.
        :type disable_search: Optional[bool]
        :param host_device: the host device for the pass.
        :type host_device: Optional[str]
        """
        assert accelerator_spec is not None, "Please specify the accelerator spec for the pass."
        assert config is not None, "Please specify the configuration for the pass."

        config_class, default_config = self.get_config_class(accelerator_spec, disable_search)

        self.accelerator_spec = accelerator_spec
        self.host_device = host_device

        self._config_class = config_class
        self.config = config
        if self._requires_user_script:
            self._user_module_loader = UserModuleLoader(self.config["user_script"], self.config["script_dir"])

        self._fixed_params = {}
        self.search_space = {}
        for k, v in self.config.items():
            if isinstance(v, SearchParameter):
                self.search_space[k] = v
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
        config = validate_config(config, config_class)

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

    def config_at_search_point(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """Get the configuration for the pass at a specific point in the search space."""
        assert set(point.keys()) == set(self.search_space.keys()), "Search point is not in the search space."
        config = self._fixed_params.copy()
        for key, value in point.items():
            config[key] = value
        return self._config_class(**config).dict()

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        """Validate the search point for the pass."""
        return True

    def run(
        self, model: OliveModelHandler, data_root: str, output_model_path: str, point: Optional[Dict[str, Any]] = None
    ) -> OliveModelHandler:
        """Run the pass on the model at a specific point in the search space."""
        point = point or {}
        config = self.config_at_search_point(point)

        if not self._initialized:
            self._initialize()
            self._initialized = True

        # Optimization pass still works on individual graphs.
        if isinstance(model, DistributedOnnxModelHandler):
            for rank in range(model.num_ranks):
                input_ranked_model = model.load_model(rank)
                ranked_output_path = Path(output_model_path).with_suffix("") / model.ranked_model_name(rank)
                self._run_for_config(input_ranked_model, data_root, config, str(ranked_output_path))

            output_model = DistributedOnnxModelHandler(
                model_path=str(Path(output_model_path).with_suffix("")),
                model_name_pattern=model.model_name_pattern,
                num_ranks=model.num_ranks,
                inference_settings=model.inference_settings,
            )
        elif isinstance(model, CompositeModelHandler) and not self._accepts_composite_model:
            # CompositePyTorchModel is also handled here.
            components = []
            component_names = []
            for component_name, component_model in model.get_model_components():
                component_output_path = Path(output_model_path).with_suffix("") / component_name
                output_model_component = self._run_for_config(
                    component_model, data_root, config, str(component_output_path)
                )
                output_model_component.model_attributes = (
                    output_model_component.model_attributes or component_model.model_attributes
                )
                components.append(output_model_component)
                component_names.append(component_name)
            output_model = CompositeModelHandler(components, component_names)
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
            "host_device": self.host_device,
            "config": self.serialize_config(self.config, check_object),
        }

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        """Pydantic validators for config params."""
        return {}

    @classmethod
    @abstractmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
                    logger.warning("Parameter %s does not have searchable values. Using default value instead.", key)
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
                if (
                    default_config[key].category == ParamCategory.PATH
                    and value is not None
                    and not isinstance(value, ResourcePath)
                ):
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

    @abstractmethod
    def _run_for_config(
        self, model: OliveModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:
        """Run the pass on the model with the given configuration."""
        raise NotImplementedError


# TODO(jambayk): rename. We are using FullPassConfig since PassConfigBase already refers to inner config
class FullPassConfig(ConfigBase):
    type: str
    disable_search: bool = False
    accelerator: Dict[str, str] = None
    host_device: Optional[str] = None
    config: Dict[str, Any] = None

    def create_pass(self):
        if not isinstance(self.accelerator, dict):
            raise ValueError(f"accelerator must be a dict, got {self.accelerator}")

        pass_cls = Pass.registry[self.type.lower()]
        accelerator_spec = AcceleratorSpec(**self.accelerator)  # pylint: disable=not-a-mapping
        return pass_cls(accelerator_spec, self.config, self.disable_search, self.host_device)


# TODO(myguo): deprecate or remove this function by explicitly specify the accelerator_spec in the arguments
# instead of using the default argument.
def create_pass_from_dict(
    pass_cls: Type[Pass],
    config: Dict[str, Any] = None,
    disable_search=False,
    accelerator_spec: AcceleratorSpec = None,
    host_device=None,
) -> Pass:
    """Create a pass from a dictionary."""
    if accelerator_spec is None:
        accelerator_spec = DEFAULT_CPU_ACCELERATOR

    config = pass_cls.generate_search_space(accelerator_spec, config, disable_search)
    return pass_cls(accelerator_spec, config, disable_search, host_device)
