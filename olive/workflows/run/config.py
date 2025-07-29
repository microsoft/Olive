# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from typing import Any, Union

from olive.auto_optimizer import AutoOptimizerConfig
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.cache import CacheConfig
from olive.common.config_utils import NestedConfig, convert_configs_to_dicts, validate_config
from olive.common.constants import DEFAULT_CACHE_DIR, DEFAULT_HF_TASK, DEFAULT_WORKFLOW_ID
from olive.common.pydantic_v1 import Field, root_validator, validator
from olive.common.utils import set_nested_dict_value
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.container.dummy_data_container import TRANSFORMER_DUMMY_DATA_CONTAINER
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine
from olive.engine.config import EngineConfig, RunPassConfig
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes.pass_config import PassParamDefault
from olive.resource_path import AZUREML_RESOURCE_TYPES
from olive.systems.common import SystemType
from olive.systems.system_config import SystemConfig


class RunEngineConfig(EngineConfig):
    evaluate_input_model: bool = Field(
        True,
        description="When true, input model will be evaluated using the engine's evaluator. Set this to false to skip.",
    )
    output_dir: Union[Path, str] = Field(None, description="Path where final output get saved.")
    packaging_config: Union[PackagingConfig, list[PackagingConfig]] = Field(
        None, description="Artifacts packaging configuration."
    )
    cache_config: Union[CacheConfig, dict[str, Any]] = Field(
        None, description="Cache configuration to speed up workflow runs."
    )
    cache_dir: Union[str, Path, list[str]] = Field(
        DEFAULT_CACHE_DIR,
        description=(
            "Path where intermediate results get saved.  Default is .olive-cache in the current working directory."
        ),
    )
    clean_cache: bool = Field(False, description="Set this to true to force clean the cache on next run.")
    clean_evaluation_cache: bool = Field(
        False, description="Set this to true to limit cache clean to generated evaluation results only."
    )
    enable_shared_cache: bool = Field(False, description="Set this to true to enable shared cache.")
    log_severity_level: int = Field(
        1,
        description=(
            "Logging level. Default is 3. Available options: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL"
        ),
    )
    ort_log_severity_level: int = Field(
        3,
        description=(
            "Log severity level for ONNX Runtime C++ logs. "
            "Default is 3. Available options: 0: VERBOSE, 1: INFO, 2: WARNING, 3: ERROR, 4: FATAL"
        ),
    )
    ort_py_log_severity_level: int = Field(
        3,
        description=(
            "Log severity level for ONNX Runtime Python logs. "
            "Available options: 0: VERBOSE, 1: INFO, 2: WARNING, 3: ERROR, 4: FATAL"
        ),
    )
    log_to_file: bool = Field(
        False,
        description=(
            "Set this to true to reroute logs to a file instead of the console. "
            "If `true`, the log will be stored in a olive-<timestamp>.log file under the current working directory."
        ),
    )

    @validator("output_dir", pre=True, always=True)
    def validate_output_dir(cls, v):
        if v is None:
            v = Path.cwd().resolve()
        else:
            v = Path(v).resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v

    def create_engine(self, olive_config, azureml_client_config, workflow_id):
        config = self.dict(include=EngineConfig.__fields__.keys())
        if self.cache_config:
            cache_config = validate_config(self.cache_config, CacheConfig)
        else:
            cache_config = CacheConfig(
                cache_dir=self.cache_dir,
                clean_cache=self.clean_cache,
                clean_evaluation_cache=self.clean_evaluation_cache,
                enable_shared_cache=self.enable_shared_cache,
            )
        return Engine(
            **config,
            olive_config=olive_config,
            cache_config=cache_config,
            azureml_client_config=azureml_client_config,
            workflow_id=workflow_id,
        )


class RunConfig(NestedConfig):
    """Run configuration for Olive workflow.

    This is the top-level configuration. It includes configurations for input model, systems, data,
    evaluators, engine, passes, and auto optimizer.
    """

    _nested_field_name = "engine"

    workflow_id: str = Field(
        DEFAULT_WORKFLOW_ID, description="Workflow ID. If not provided, use the default ID 'default_workflow'."
    )
    azureml_client: AzureMLClientConfig = Field(
        None,
        description=(
            "AzureML client configuration. This client configuration will be used for all AzureML related resources in"
            " the workflow."
        ),
    )
    input_model: ModelConfig = Field(description="Input model configuration.")
    systems: dict[str, SystemConfig] = Field(
        None,
        description="System configurations. Other fields such as engine and passes can refer to these systems by name.",
    )
    data_configs: list[DataConfig] = Field(
        default_factory=list,
        description=(
            "Data configurations. Each data config must have a unique name. Other fields such as engine, passes and"
            " evaluators can refer to these data configs by name. In auto-optimizer mode, only one data config is"
            " allowed."
        ),
    )
    evaluators: dict[str, OliveEvaluatorConfig] = Field(
        None,
        description=(
            "Evaluator configurations. Other fields such as engine and passes can refer to these evaluators by name."
        ),
    )
    engine: RunEngineConfig = Field(
        default_factory=RunEngineConfig,
        description=(
            "Engine configuration. If not provided, the workflow uses the default engine configuration which runs in"
            " no-search or auto-optimizer mode based on whether passes field is provided."
        ),
    )
    passes: dict[str, list[RunPassConfig]] = Field(default_factory=dict, description="Pass configurations.")
    auto_optimizer_config: AutoOptimizerConfig = Field(
        default_factory=AutoOptimizerConfig,
        description="Auto optimizer configuration. Only valid when passes field is empty or not provided.",
    )
    workflow_host: SystemConfig = Field(
        None, description="Workflow host. None by default. If provided, the workflow will be run on the specified host."
    )

    @root_validator(pre=True)
    def patch_evaluators(cls, values):
        if "evaluators" in values:
            for name, evaluator_config in values["evaluators"].items():
                evaluator_config["name"] = name
        return values

    @root_validator(pre=True)
    def patch_passes(cls, values):
        if "passes" in values:
            for name, passes_config in values["passes"].items():
                if isinstance(passes_config, dict):
                    values["passes"][name] = [passes_config]
        return values

    @root_validator(pre=True)
    def insert_azureml_client(cls, values):
        values = convert_configs_to_dicts(values)
        _insert_azureml_client(values, values.get("azureml_client"))
        return values

    @root_validator()
    def validate_python_environment_paths(cls, values):
        # Check if we need to validate python environment path
        engine = values.get("engine")
        if engine:
            engine_host = engine.host
            if not engine_host or engine_host.type != SystemType.Docker:
                systems = values.get("systems")
                if systems:
                    _validate_python_environment_path(systems)
        return values

    @validator("data_configs", pre=True)
    def validate_data_config_names(cls, v):
        if not v:
            return v

        # validate data config name is unique
        data_name_set = set()
        for data_config in v:
            data_config_obj = validate_config(data_config, DataConfig)
            if data_config_obj.name in data_name_set:
                raise ValueError(f"Data config name {data_config_obj.name} is duplicated. Please use another name.")
            data_name_set.add(data_config_obj.name)
        return v

    @validator("data_configs", pre=True, each_item=True)
    def validate_data_configs_with_hf_model(cls, v, values):
        if "input_model" not in values:
            raise ValueError("Invalid input model")

        input_model_config = values["input_model"].dict()
        if input_model_config["type"].lower() not in ["hfmodel", "onnxmodel"]:
            return v

        if isinstance(v, DataConfig):
            v = v.dict()

        # all model related info used for auto filling
        task = None
        model_name = None
        if input_model_config["type"].lower() == "onnxmodel" and "model_attributes" in input_model_config["config"]:
            # Only valid for onnx model converted from Olive Conversion pass
            task = input_model_config["config"]["model_attributes"].get("hf_task", DEFAULT_HF_TASK)
            model_name = input_model_config["config"]["model_attributes"].get("_name_or_path")
        else:
            task = input_model_config["config"].get("task", DEFAULT_HF_TASK)
            model_name = input_model_config["config"]["model_path"]

        model_info = {
            "model_name": model_name,
            "task": task,
            "trust_remote_code": input_model_config["config"].get("load_kwargs", {}).get("trust_remote_code"),
        }
        kv_cache = input_model_config.get("io_config", {}).get("kv_cache")
        if isinstance(kv_cache, dict):
            for key in ["ort_past_key_name", "ort_past_value_name", "batch_size"]:
                model_info[key] = kv_cache.get(key)

        # TODO(anyone): Will this container ever be used with non-HF models?
        if v.get("type"):
            if v["type"] in TRANSFORMER_DUMMY_DATA_CONTAINER:
                _auto_fill_data_config(
                    v,
                    model_info,
                    ["load_dataset_config"],
                    ["model_name", "ort_past_key_name", "ort_past_value_name", "batch_size"],
                )
            elif v["type"] == HuggingfaceContainer.__name__:
                # auto insert model_name and task from input model hf config if not present
                # both are required for huggingface container
                _auto_fill_data_config(
                    v, model_info, ["pre_process_data_config", "post_process_data_config"], ["model_name", "task"]
                )

                # auto insert trust_remote_code from input model hf config
                # won't override if value was set to False explicitly
                _auto_fill_data_config(
                    v,
                    model_info,
                    ["pre_process_data_config", "load_dataset_config"],
                    ["trust_remote_code"],
                    only_none=True,
                )

        return validate_config(v, DataConfig)

    @validator("evaluators", pre=True, each_item=True)
    def validate_evaluators(cls, v, values):
        for idx, metric in enumerate(v.get("metrics", [])):
            v["metrics"][idx] = _resolve_data_config(metric, values, "data_config")
        return v

    @validator("auto_optimizer_config", pre=True)
    def validate_auto_optimizer_config(cls, v, values):
        _resolve_all_data_configs(v, values)
        return v

    @validator("engine", pre=True)
    def validate_engine(cls, v, values):
        v = _resolve_system(v, values, "host")
        v = _resolve_system(v, values, "target")
        if v.get("search_strategy") and not v.get("evaluator"):
            raise ValueError(
                "Can't search without a valid evaluator config. "
                "Either provider a valid evaluator config or disable search."
            )

        return _resolve_evaluator(v, values)

    @validator("passes", pre=True, each_item=True)
    def validate_pass_host_evaluator(cls, v, values):
        for i, _ in enumerate(v):
            v[i] = _resolve_system(v[i], values, "host")
            v[i] = _resolve_evaluator(v[i], values)
        return v

    @validator("passes", pre=True, each_item=True)
    def validate_pass_search(cls, v, values):
        if "engine" not in values:
            raise ValueError("Invalid engine")

        for i, _ in enumerate(v):
            # validate first to gather config params
            v[i] = iv = validate_config(v[i], RunPassConfig).dict()

            if iv.get("config"):
                _resolve_all_data_configs(iv["config"], values)

                searchable_configs = set()
                for param_name in iv["config"]:
                    if iv["config"][param_name] == PassParamDefault.SEARCHABLE_VALUES:
                        searchable_configs.add(param_name)

                if not values["engine"].search_strategy and searchable_configs:
                    raise ValueError(
                        f"You cannot disable search for {iv['type']} and"
                        f" set {searchable_configs} to SEARCHABLE_VALUES at the same time."
                        " Please remove SEARCHABLE_VALUES or enable search (needs search strategy configs)."
                    )
        return v

    @validator("workflow_host", pre=True)
    def validate_workflow_host(cls, v, values):
        return _resolve_config(values, v)


def _validate_python_environment_path(systems):
    for system_config in systems.values():
        if system_config.type != SystemType.PythonEnvironment:
            continue

        python_environment_path = system_config.config.python_environment_path
        if python_environment_path is None:
            raise ValueError("python_environment_path is required for PythonEnvironmentSystem native mode")

        # check if the path exists
        if not Path(python_environment_path).exists():
            raise ValueError(f"Python path {python_environment_path} does not exist")

        # check if python exists in the path
        python_path = shutil.which("python", path=python_environment_path)
        if not python_path:
            raise ValueError(f"Python executable not found in the path {python_environment_path}")


def _resolve_all_data_configs(config, values):
    """Recursively traverse the config dictionary to resolve all 'data_config' keys."""
    if isinstance(config, dict):
        for param_name, param_value in config.items():
            if param_name.endswith("data_config"):
                _resolve_data_config(config, values, param_name)
            else:
                _resolve_all_data_configs(param_value, values)
    elif isinstance(config, list):
        for element in config:
            _resolve_all_data_configs(element, values)


def _insert_azureml_client(config, azureml_client):
    """Insert azureml_client into config recursively.

    Valid cases:
        1. AzureML resource path config without azureml_client
        2. AzureML system config without azureml_client_config
    config is modified in place.
    """
    if not isinstance(config, (dict, list)):
        return

    insert_key, config_type = _needs_aml_client(config)

    if insert_key and not azureml_client:
        raise ValueError(f"azureml_client is required for {config_type} but not provided.")
    elif insert_key:
        set_nested_dict_value(config, insert_key, azureml_client)
        return

    for value in config.values() if isinstance(config, dict) else config:
        _insert_azureml_client(value, azureml_client)


def _needs_aml_client(config):
    """Check if azureml_client is needed for the given config.

    Return the path to insert azureml_client and the type of the config.
    """
    if not isinstance(config, dict):
        return None, None

    config_type = config.get("type")

    support_types = {
        "AzureML Resource Path": {
            "types": AZUREML_RESOURCE_TYPES,
            "param": "azureml_client",
        },
        "AzureML System": {
            "types": ["AzureML"],
            "param": "azureml_client_config",
        },
    }
    for type_name, type_info in support_types.items():
        if config_type not in type_info["types"]:
            continue

        # check if azureml_client is already provided
        # it could be provided in the config or in the config's config
        if config.get(type_info["param"]) or config.get("config", {}).get(type_info["param"]):
            return None, None

        # will directly insert azureml_client into the config
        return (type_info["param"],), type_name

    return None, None


def _resolve_config_str(v, values, alias, component_name):
    """Resolve string value for alias in v to corresponding component config in values.

    values: {
        ...
        component_name: {
            ...
            component_name_1: component_config_1,
            ...
        }
        ...
    }

    v: {
        ...
        alias: component_name_1
        ...
    }
    -> {
        ...
        alias: component_config_1
        ...
    }
    """
    if not isinstance(v, dict):
        # if not a dict, return the original value
        return v

    # get name of sub component
    sub_component = v.get(alias)
    if not isinstance(sub_component, str):
        return v

    component_config = _resolve_config(values, sub_component, component_name)

    v[alias] = component_config
    return v


def _resolve_config(values, sub_component, component_name="systems"):
    # resolve component name to component configs
    if component_name not in values:
        raise ValueError(f"Invalid {component_name}")

    components = values[component_name] or {}
    # resolve sub component name to component config
    if sub_component not in components:
        raise ValueError(f"{sub_component} not found in {components}")
    return components[sub_component]


def _resolve_system(v, values, system_alias):
    return _resolve_config_str(v, values, system_alias, component_name="systems")


def _resolve_data_config(v, values, data_config_alias):
    if isinstance(data_config_alias, (dict, DataConfig)):
        raise ValueError(
            "Inline data configs are not supported. Define the config under 'data_configs' and use its name here."
        )

    return _resolve_config_str(
        v,
        {"data_configs_dict": {dc.name: dc for dc in values.get("data_configs") or []}},
        data_config_alias,
        component_name="data_configs_dict",
    )


def _resolve_evaluator(v, values):
    if not isinstance(v, dict):
        return v

    evaluator = v.get("evaluator")
    if isinstance(evaluator, dict):
        for idx, metric in enumerate(evaluator.get("metrics", [])):
            evaluator["metrics"][idx] = _resolve_data_config(metric, values, "data_config")
        return v

    return _resolve_config_str(v, values, "evaluator", component_name="evaluators")


def _auto_fill_data_config(config, info, config_names, param_names, only_none=False):
    """Auto fill data config with model info.

    :param config: data config
    :param info: model info
    :param config_names: list of config names to fill
    :param param_names: list of param names to fill in each config
    :param only_none: only fill if the value is None, otherwise fill if the value is falsy
    """
    for component_config_name in config_names:
        # validate the component config first to gather the config params
        config[component_config_name] = component_config = validate_config(
            config.get(component_config_name) or {}, DataComponentConfig
        ).dict()
        component_config["params"] = component_config_params = component_config.get("params") or {}

        for key in param_names:
            if info.get(key) is None:
                continue

            if (only_none and component_config_params.get(key) is None) or (
                not only_none and not component_config_params.get(key)
            ):
                component_config_params[key] = info[key]
