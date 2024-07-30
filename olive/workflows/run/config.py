# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, List, Union

from olive.auto_optimizer import AutoOptimizerConfig
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import NestedConfig, convert_configs_to_dicts, validate_config
from olive.common.constants import DEFAULT_HF_TASK, DEFAULT_WORKFLOW_ID
from olive.common.pydantic_v1 import Field, root_validator, validator
from olive.common.utils import set_nested_dict_value
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.container.dummy_data_container import TRANSFORMER_DUMMY_DATA_CONTAINER
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine, EngineConfig
from olive.engine.cloud_cache_helper import CloudCacheConfig
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes import AbstractPassConfig
from olive.passes.pass_config import PassParamDefault
from olive.resource_path import AZUREML_RESOURCE_TYPES
from olive.systems.system_config import SystemConfig

logger = logging.getLogger(__name__)


class RunPassConfig(AbstractPassConfig):
    """Pass configuration for Olive workflow.

    This is the configuration for a single pass in Olive workflow. It includes configurations for pass type, config,
    etc.

    Example:
    .. code-block:: json

        {
            "type": "OlivePass",
            "config": {
                "param1": "value1",
                "param2": "value2"
            }
        }

    """

    host: Union[SystemConfig, str] = Field(
        None,
        description=(
            "Host system for the pass. If it is a string, must refer to a system config under `systems` section. If not"
            " provided, use the engine's host system."
        ),
    )
    evaluator: Union[OliveEvaluatorConfig, str] = Field(
        None,
        description=(
            "Evaluator for the pass. If it is a string, must refer to an evaluator config under `evaluators` section."
            " If not provided, use the engine's evaluator."
        ),
    )
    clean_run_cache: bool = Field(
        False,
        description=(
            "Whether to clean the run cache before running the pass. If set to True, the cache related to all runs"
            " related to this pass type will be cleaned."
        ),
    )
    output_name: str = Field(
        None,
        description=(
            "Prefix of the name of the output of this pass in the output directory. Only used when the workflow is run"
            " in no-search mode. If not provided, only the output of the last pass will be saved."
        ),
    )


class RunEngineConfig(EngineConfig):
    evaluate_input_model: bool = True
    output_dir: Union[Path, str] = None
    output_name: str = None
    packaging_config: Union[PackagingConfig, List[PackagingConfig]] = None
    cloud_cache_config: Union[bool, CloudCacheConfig] = False
    log_severity_level: int = 1
    ort_log_severity_level: int = 3
    ort_py_log_severity_level: int = 3
    log_to_file: bool = False

    @validator("cloud_cache_config", pre=True, always=True)
    def validate_cloud_cache(cls, v):
        if isinstance(v, bool):
            cloud_cache_config = CloudCacheConfig()
            cloud_cache_config.enable_cloud_cache = v
            v = cloud_cache_config
        return v

    def create_engine(self, azureml_client_config, workflow_id):
        config = self.dict(include=EngineConfig.__fields__.keys())
        return Engine(**config, azureml_client_config=azureml_client_config, workflow_id=workflow_id)


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
    systems: Dict[str, SystemConfig] = Field(
        None,
        description="System configurations. Other fields such as engine and passes can refer to these systems by name.",
    )
    data_configs: List[DataConfig] = Field(
        default_factory=list,
        description=(
            "Data configurations. Each data config must have a unique name. Other fields such as engine, passes and"
            " evaluators can refer to these data configs by name. In auto-optimizer mode, only one data config is"
            " allowed."
        ),
    )
    evaluators: Dict[str, OliveEvaluatorConfig] = Field(
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
    passes: Dict[str, RunPassConfig] = Field(None, description="Pass configurations.")
    pass_flows: List[List[str]] = Field(
        None,
        description=(
            "Pass flows. Each member must be a list of pass names from `passes` field. If provided,"
            " each flow will be run sequentially. If not provided, all passes will be run as a single flow in the order"
            " of `passes` field."
        ),
    )
    auto_optimizer_config: AutoOptimizerConfig = Field(
        default_factory=AutoOptimizerConfig,
        description="Auto optimizer configuration. Only valid when passes field is empty or not provided.",
    )
    workflow_host: SystemConfig = Field(
        None, description="Workflow host. None by default. If provided, the workflow will be run on the specified host."
    )

    @root_validator(pre=True)
    def insert_azureml_client(cls, values):
        values = convert_configs_to_dicts(values)
        _insert_azureml_client(values, values.get("azureml_client"))

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
        if input_model_config["type"].lower() != "hfmodel":
            return v

        if isinstance(v, DataConfig):
            v = v.dict()

        # all model related info used for auto filling
        model_info = {
            "model_name": input_model_config["config"]["model_path"],
            "task": input_model_config["config"].get("task", DEFAULT_HF_TASK),
        }
        kv_cache = input_model_config.get("io_config", {}).get("kv_cache")
        if isinstance(kv_cache, dict):
            for key in ["ort_past_key_name", "ort_past_value_name", "batch_size"]:
                model_info[key] = kv_cache.get(key)
        if input_model_config["config"].get("load_kwargs", {}).get("trust_remote_code"):
            model_info["trust_remote_code"] = True

        # TODO(anyone): Will this container ever be used with non-HF models?
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

    @validator("engine", pre=True)
    def validate_engine(cls, v, values):
        v = _resolve_system(v, values, "host")
        v = _resolve_system(v, values, "target")
        return _resolve_evaluator(v, values)

    @validator("engine")
    def validate_evaluate_input_model(cls, v):
        if v.evaluate_input_model and v.evaluator is None:
            logger.info("No evaluator is specified, skip to evaluate model")
            v.evaluate_input_model = False
        return v

    @validator("passes", pre=True, each_item=True)
    def validate_pass_host_evaluator(cls, v, values):
        v = _resolve_system(v, values, "host")
        return _resolve_evaluator(v, values)

    @validator("passes", pre=True, each_item=True)
    def validate_pass_search(cls, v, values):
        if "engine" not in values:
            raise ValueError("Invalid engine")

        disable_search = v.get("disable_search")
        if not values["engine"].search_strategy:
            if disable_search is False:
                raise ValueError("You cannot set disable_search is False if search strategy is None/False/{}")
            # disable search if search_strategy is None/False/{}, user cannot override it.
            # If user explicitly set, raise error when disable_search is False and search_strategy is None/False/{}
            if disable_search is None:
                disable_search = True
        else:
            # disable search if user explicitly set disable_search to True
            disable_search = disable_search or False

        v["disable_search"] = disable_search

        # validate first to gather config params
        v = validate_config(v, RunPassConfig).dict()

        if not v.get("config"):
            return v

        searchable_configs = set()
        for param_name in v["config"]:
            if v["config"][param_name] == PassParamDefault.SEARCHABLE_VALUES:
                searchable_configs.add(param_name)
            if param_name.endswith("data_config"):
                v["config"] = _resolve_data_config(v["config"], values, param_name)

        if disable_search and searchable_configs:
            raise ValueError(
                f"You cannot disable search for {v['type']} and"
                f" set {searchable_configs} to SEARCHABLE_VALUES at the same time."
                " Please remove SEARCHABLE_VALUES or enable search(needs search strategy configs)."
            )
        return v

    @validator("workflow_host", pre=True)
    def validate_workflow_host(cls, v, values):
        if v is None:
            return v
        return _resolve_config(values, v)


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
