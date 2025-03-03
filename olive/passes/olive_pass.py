# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union, get_args

from olive.common.config_utils import ParamCategory, validate_config
from olive.common.pydantic_v1 import BaseModel, ValidationError, create_model
from olive.common.user_module_loader import UserModuleLoader
from olive.data.config import DataConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec
from olive.model import CompositeModelHandler, DistributedOnnxModelHandler, OliveModelHandler, ONNXModelHandler
from olive.passes.pass_config import (
    AbstractPassConfig,
    BasePassConfig,
    PassConfigParam,
    PassParamDefault,
    create_config_class,
)
from olive.resource_path import ResourcePath
from olive.search.search_parameter import (
    Categorical,
    Conditional,
    ConditionalDefault,
    SearchParameter,
    SpecialParamValue,
)
from olive.search.utils import cyclic_search_space, order_search_parameters

logger = logging.getLogger(__name__)

# ruff: noqa: B027


class Pass(ABC):
    """Base class for pass configuration.

    Each pass should derive its own configuration class that contains all information it needs to execute.
    """

    registry: ClassVar[Dict[str, Type["Pass"]]] = {}
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
        self,
        accelerator_spec: AcceleratorSpec,
        config: Type[BasePassConfig],
        host_device=None,
    ):
        """Initialize the pass.

        :param accelerator_spec: the accelerator spec for the pass.
        :type accelerator_spec: AcceleratorSpec
        :param config: the configuration representing search space.
        :type config: Type[BasePassConfig]
        :param host_device: the host device for the pass.
        :type host_device: Optional[str]
        """
        assert accelerator_spec is not None, "Please specify the accelerator spec for the pass."
        assert config is not None, "Please specify the configuration for the pass."

        self.config = config
        self.accelerator_spec = accelerator_spec
        self.host_device = host_device

        if hasattr(self.config, "user_script") and hasattr(self.config, "script_dir"):
            self._user_module_loader = UserModuleLoader(self.config.user_script, self.config.script_dir)

        # Params that are paths [(param_name, required)]
        self.path_params = [
            (param, param_config.required, param_config.category)
            for param, param_config in self.default_config(accelerator_spec).items()
            if param_config.category in (ParamCategory.PATH, ParamCategory.DATA)
        ]

        self._initialized = False

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Whether the pass is accelerator agnostic. If True, the pass will be reused for all accelerators.

        The default value is True. The subclass could choose to override this method to return False by using the
        accelerator spec information.
        """
        return True

    @classmethod
    def get_config_params(
        cls,
        accelerator_spec: AcceleratorSpec,
        config: Optional[Dict[str, Any]] = None,
        disable_search: Optional[bool] = False,
    ) -> Tuple[Type[BasePassConfig], Dict[str, Any], Dict[str, SearchParameter]]:
        """Generate search space for the pass."""
        assert accelerator_spec is not None, "Please specify the accelerator spec for the pass"
        config = config or {}

        # Get the config class with default value or default search value
        config_class, default_config = cls.get_config_class(accelerator_spec, disable_search)

        if not disable_search:
            # Replace user-provided values with Categorical if user intended to search
            config = cls._identify_search_values(config, default_config)

        # Generate the search space by using both default value and default search value and user provided config
        config = validate_config(config, config_class)
        config = cls._resolve_config(config, default_config)
        return config_class, *cls._init_fixed_and_search_params(config, default_config)

    @classmethod
    def generate_config(
        cls,
        accelerator_spec: AcceleratorSpec,
        config: Optional[Dict[str, Any]] = None,
        point: Optional[Dict[str, Any]] = None,
        disable_search: Optional[bool] = False,
    ) -> Type[BasePassConfig]:
        """Get the configuration for the pass at a specific point in the search space."""
        assert accelerator_spec is not None, "Please specify the accelerator spec for the pass"

        point = point or {}
        config_class, fixed_values, search_params = cls.get_config_params(accelerator_spec, config, disable_search)
        assert (
            set(point.keys()).intersection(set(search_params.keys())) == point.keys()
        ), "Search point is not in the search space."
        return config_class.parse_obj({**fixed_values, **search_params, **point})

    @classmethod
    def _identify_search_values(
        cls,
        config: Dict[str, Any],
        default_config: Dict[str, PassConfigParam],
    ):
        """Conditionally, replace user provided search values with Categorical."""
        for name, param in default_config.items():
            if param.search_defaults and name in config:
                value = config[name]

                # If the user provided a non-empty list, validate if "type of
                # each elements" in the list is the same as the "param's expected type".
                # If successful, treat it as a searchable value i.e. turn it into a Categorical.
                if isinstance(value, list) and len(value) > 0:
                    dummy_values_config = {name: (List[param.type_], None)}
                    dummy_values_model = create_model(
                        f"SearchableParamConfig_{name}_values", **dummy_values_config, __base__=BaseModel
                    )

                    try:
                        validate_config({name: value}, dummy_values_model)
                        config[name] = Categorical(value)
                        continue
                    except ValidationError:
                        # Expected in certain cases and intentionally ignored!!
                        pass

                # If not, leave the value alone so that the default validation
                # would report an appropriate error.

        return config

    @classmethod
    def get_config_class(cls, accelerator_spec: AcceleratorSpec, disable_search: Optional[bool] = False):
        """Get the configuration class for the pass."""
        default_config = cls.default_config(accelerator_spec)
        config_class = create_config_class(cls.__name__, default_config, disable_search, cls._validators())
        return config_class, default_config

    @classmethod
    def default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        """Get the default configuration for the pass."""
        config = cls._default_config(accelerator_spec)
        # validate that all parameters ending with data_config are of type DataConfig, Union[DataConfig, dict], ...
        # this requirement is on the pass developer but we can only check it here
        for param, param_config in config.items():
            if param.endswith("data_config"):
                param_type = param_config.type_
                assert param_type == DataConfig or DataConfig in get_args(
                    param_type
                ), f"{param} ending with data_config must be of type DataConfig."
        return config

    @classmethod
    def validate_config(
        cls,
        config: Type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        """Validate the input config for the pass."""
        return True

    def run(self, model: OliveModelHandler, output_model_path: str) -> OliveModelHandler:
        """Run the pass on the model at a specific point in the search space."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

        # Optimization pass still works on individual graphs.
        if isinstance(model, DistributedOnnxModelHandler):
            for rank in range(model.num_ranks):
                input_ranked_model = model.load_model(rank)
                ranked_output_path = Path(output_model_path).with_suffix("") / model.ranked_model_name(rank)
                self._run_for_config(input_ranked_model, self.config, str(ranked_output_path))

            # ranked model don't have their own model_attributes, they are just part of the distributed model
            # which has the model_attributes
            output_model = DistributedOnnxModelHandler(
                model_path=str(Path(output_model_path).with_suffix("")),
                model_name_pattern=model.model_name_pattern,
                num_ranks=model.num_ranks,
                inference_settings=model.inference_settings,
                model_attributes=model.model_attributes,
            )
            Pass._carry_forward_additional_files(model, output_model)
        elif isinstance(model, CompositeModelHandler) and not self._accepts_composite_model:
            components = []
            component_names = []
            for component_name, component_model in model.get_model_components():
                component_output_path = Path(output_model_path).with_suffix("") / component_name
                output_model_component = self._run_for_config(component_model, self.config, str(component_output_path))
                output_model_component.model_attributes = (
                    output_model_component.model_attributes or component_model.model_attributes
                )
                components.append(output_model_component)
                component_names.append(component_name)
                Pass._carry_forward_additional_files(component_model, output_model_component)
            output_model = CompositeModelHandler(components, component_names)
            output_model.model_attributes = output_model.model_attributes or model.model_attributes
        else:
            output_model = self._run_for_config(model, self.config, output_model_path)
            # assumption: the model attributes from passes, if any, are more important than
            # the input model attributes, we should not update/extend anymore outside of the pass run
            output_model.model_attributes = output_model.model_attributes or model.model_attributes
            if not isinstance(output_model, CompositeModelHandler):
                # save and carry forward additional files into the the output model path
                # for composite model, the additional_files attribute is already present in the parent
                # model_attributes
                Pass._carry_forward_additional_files(model, output_model)

        return output_model

    @staticmethod
    def _carry_forward_additional_files(input_model: OliveModelHandler, output_model: OliveModelHandler):
        # NOTE: Can't use model.model_path because that always gets resolved to a filepath.
        # We need the directory path here.
        input_model_path = input_model.get_resource("model_path")
        if not input_model_path:
            return

        input_model_path = Path(input_model_path)
        if not input_model_path.is_dir():
            return

        input_model_attributes = input_model.model_attributes or {}
        input_model_additional_files = set(input_model_attributes.get("additional_files", []))
        if not input_model_additional_files:
            return

        output_model_path = Path(output_model.get_resource("model_path"))
        if not output_model_path.is_dir():
            if isinstance(output_model, ONNXModelHandler):
                # change the "model_path" resource to the parent directory of the model file
                output_model_path = output_model.change_model_path_to_dir()
            else:
                logger.warning("Expecting the output model to be in a directory but found a file.")
                return

        output_model_attributes = output_model.model_attributes or {}
        # output model might have inherited model_attributes from input model
        # remove the input model's additional files from the output model's additional files
        # we will add the files that are not already present in the output model
        output_model_additional_files = (
            set(output_model_attributes.get("additional_files", [])) - input_model_additional_files
        )

        for filepath in input_model_additional_files:
            input_filepath = Path(filepath)

            # Make sure we don't overwrite an existing file in the output's directory.
            # The follow up pass could have *potentially* generated a file with the same name.
            output_filepath = output_model_path / input_filepath.name
            if not output_filepath.exists():
                # TODO(team): Use symlinks instead of copying the files.
                shutil.copy(str(input_filepath), str(output_filepath))
            # always add the file_path to the output model's additional files
            # this covers the case where the output model_path is the same as the input model_path
            # like for perf-tuning pass
            output_model_additional_files.add(str(output_filepath))

        output_model_attributes["additional_files"] = sorted(output_model_additional_files)
        output_model.model_attributes = output_model_attributes

    def to_json(self, check_object: bool = False) -> Dict[str, Any]:
        """Convert the pass to json."""
        return {
            "type": self.__class__.__name__,
            "accelerator": self.accelerator_spec.to_json(),
            "host_device": self.host_device,
            "config": self.config.to_json(check_object),
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
            ```
            return {
                # required parameter
                "param1": PassConfigParam(type_=int, required=True, description="param1 description"),
                # optional parameter with default value
                "param2": PassConfigParam(type_=int, default_value=1, description="param2 description"),
                # optional parameter with default value and searchable values
                "param3": PassConfigParam(
                    type_=int,
                    default_value=1,
                    search_defaults=Categorical([1, 2, 3]),
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
                # optional parameter with search_defaults that depends on other parameter values
                "param6": PassConfigParam(
                    type_=int,
                    default_value=1,
                    search_defaults=Conditional(
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
            ```

        """
        raise NotImplementedError

    @classmethod
    def _resolve_defaults(cls, config: Dict[str, Any], default_config: Dict[str, PassConfigParam]) -> Dict[str, Any]:
        """Resolve default values."""
        for key, value in config.items():
            if value == PassParamDefault.DEFAULT_VALUE:
                config[key] = default_config[key].default_value
            elif value == PassParamDefault.SEARCHABLE_VALUES:
                v = default_config[key].search_defaults
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
        return fixed_params, search_space

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
        input_config: Union[Dict[str, Any], Type[BasePassConfig]],
        default_config: Dict[str, PassConfigParam],
    ) -> Dict[str, Any]:
        """Resolve config to BasePassConfig."""
        config = input_config.dict()
        config = cls._resolve_defaults(config, default_config)
        if "user_script" in config:
            user_module_loader = UserModuleLoader(config["user_script"], config["script_dir"])
            config = cls._validate_user_script(config, user_module_loader, default_config)
        return config

    def _initialize(self):
        """Initialize the pass. Pass specific initialization should be done here."""

    @abstractmethod
    def _run_for_config(
        self, model: OliveModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> OliveModelHandler:
        """Run the pass on the model with the given configuration."""
        raise NotImplementedError


class FullPassConfig(AbstractPassConfig):
    """Configuration for a pass serialization.

    This class can be used to serialize a pass configuration to a JSON file and
    reconstruct the pass from the JSON file.
    """

    accelerator: Dict[str, str] = None
    host_device: Optional[str] = None

    def create_pass(self):
        if not isinstance(self.accelerator, dict):
            raise ValueError(f"accelerator must be a dict, got {self.accelerator}")

        pass_cls = Pass.registry[self.type.lower()]
        accelerator_spec = AcceleratorSpec(**self.accelerator)  # pylint: disable=not-a-mapping
        self.config = pass_cls.generate_config(accelerator_spec, self.config)
        return pass_cls(accelerator_spec, self.config, self.host_device)


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

    config: Type[BasePassConfig] = pass_cls.generate_config(accelerator_spec, config, disable_search=disable_search)
    return pass_cls(accelerator_spec, config, host_device)
