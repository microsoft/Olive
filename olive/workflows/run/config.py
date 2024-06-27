# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, List, Union

from olive.auto_optimizer import AutoOptimizerConfig
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, validate_config
from olive.common.constants import DEFAULT_WORKFLOW_ID
from olive.common.pydantic_v1 import Field, validator
from olive.data.config import DataConfig
from olive.data.container.dummy_data_container import TransformersDummyDataContainer
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine, EngineConfig
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
    log_severity_level: int = 1
    ort_log_severity_level: int = 3
    ort_py_log_severity_level: int = 3
    log_to_file: bool = False

    def create_engine(self, azureml_client_config, workflow_id):
        config = self.dict(include=EngineConfig.__fields__.keys())
        return Engine(**config, azureml_client_config=azureml_client_config, workflow_id=workflow_id)


class RunConfig(ConfigBase):
    """Run configuration for Olive workflow.

    This is the top-level configuration. It includes configurations for input model, systems, data,
    evaluators, engine, passes, and auto optimizer.
    """

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
    data_root: str = Field(
        None,
        description=(
            "Root directory for data. If provided, all relative data paths in other configs will be resolved based on"
            " this root."
        ),
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

    @validator("input_model", pre=True)
    def insert_aml_client(cls, v, values):
        if isinstance(v, ModelConfig):
            v = v.dict()

        input_model_path = v["config"].get("model_path")
        # if not a dict, return the original value
        if not input_model_path or not isinstance(input_model_path, dict):
            return v

        if _have_aml_client(input_model_path, values):
            v["config"]["model_path"]["config"]["azureml_client"] = values["azureml_client"]
        return v

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
    def validate_data_configs(cls, v, values):
        if "input_model" not in values:
            raise ValueError("Invalid input model")

        if isinstance(v, DataConfig):
            v = v.dict()

        if v["type"] == TransformersDummyDataContainer.__name__:
            # if user already set kv_cache in io_config, use it directly and ignore the default value
            io_config = values["input_model"].dict()["config"].get("io_config", {})
            hf_config = values["input_model"].dict()["config"].get("hf_config", {})
            kv_cache = io_config.get("kv_cache", False)

            for component_config_name in ["load_dataset_config"]:
                v[component_config_name] = component_config = v.get(component_config_name) or {}
                component_config["params"] = component_config_params = component_config.get("params") or {}
                for key in ["model_name"]:
                    if not component_config_params.get(key, None):
                        component_config_params[key] = hf_config.get(key, None)
                if isinstance(kv_cache, dict):
                    for key in ["ort_past_key_name", "ort_past_value_name", "batch_size"]:
                        if not component_config_params.get(key, None) and kv_cache.get(key):
                            component_config_params[key] = kv_cache.get(key, None)

        if v["type"] == HuggingfaceContainer.__name__:
            hf_config = values["input_model"].dict()["config"].get("hf_config", {})

            # auto insert model_name and task from input model hf config if not present
            # both are required for huggingface container
            for component_config_name in ["load_dataset_config", "pre_process_data_config", "post_process_data_config"]:
                v[component_config_name] = component_config = v.get(component_config_name) or {}
                component_config["params"] = component_config_params = component_config.get("params") or {}
                for key in ["model_name", "task"]:
                    if not component_config_params.get(key, None):
                        component_config_params[key] = hf_config.get(key, None)

            # auto insert trust_remote_code from input model hf config
            # won't override if value was set to False explicitly
            if hf_config.get("from_pretrained_args", {}).get("trust_remote_code"):
                for config_name in ["pre_process_data_config", "load_dataset_config"]:
                    v[config_name] = component_config = v.get(config_name) or {}
                    component_config["params"] = component_config_params = component_config.get("params") or {}
                    if component_config_params.get("trust_remote_code") is None:
                        component_config_params["trust_remote_code"] = hf_config["from_pretrained_args"][
                            "trust_remote_code"
                        ]

        return validate_config(v, DataConfig)

    @validator("evaluators", pre=True, each_item=True)
    def validate_evaluators(cls, v, values):
        for idx, metric in enumerate(v.get("metrics", [])):
            v["metrics"][idx] = _resolve_data_config(metric, values, "data_config")

            data_dir_config = v["metrics"][idx].get("user_config", {}).get("data_dir", None)
            if isinstance(data_dir_config, dict):
                if _have_aml_client(data_dir_config, values):
                    data_dir_config["config"]["azureml_client"] = values["azureml_client"]
                v["metrics"][idx]["user_config"]["data_dir"] = data_dir_config
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
        if not v.get("config"):
            return v

        searchable_configs = set()
        for param_name in v["config"]:
            if v["config"][param_name] == PassParamDefault.SEARCHABLE_VALUES:
                searchable_configs.add(param_name)
            if param_name.endswith("data_config"):
                v["config"] = _resolve_data_config(v["config"], values, param_name)

        data_dir_config = v["config"].get("data_dir", None)
        if isinstance(data_dir_config, dict):
            if _have_aml_client(data_dir_config, values):
                data_dir_config["config"]["azureml_client"] = values["azureml_client"]
            v["config"]["data_dir"] = data_dir_config

        if disable_search and searchable_configs:
            raise ValueError(
                f"You cannot disable search for {v['type']} and"
                f" set {searchable_configs} to SEARCHABLE_VALUES at the same time."
                " Please remove SEARCHABLE_VALUES or enable search(needs search strategy configs)."
            )
        return v


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

    # resolve component name to component configs
    if component_name not in values:
        raise ValueError(f"Invalid {component_name}")
    components = values[component_name] or {}
    # resolve sub component name to component config
    if sub_component not in components:
        raise ValueError(f"{alias} {sub_component} not found in {components}")
    v[alias] = components[sub_component]
    return v


def _resolve_system(v, values, system_alias):
    v = _resolve_config_str(v, values, system_alias, component_name="systems")
    if v.get(system_alias):
        v[system_alias] = validate_config(v[system_alias], SystemConfig)
        if v[system_alias].type == "AzureML":
            if not values["azureml_client"]:
                raise ValueError("AzureML client config is required for AzureML system")
            v[system_alias].config.azureml_client_config = values["azureml_client"]
    return v


def _resolve_data_config(v, values, data_config_alias):
    if not isinstance(v, dict):
        # if not a dict, return the original value
        return v

    # get name of sub component
    sub_component = v.get(data_config_alias)
    if not isinstance(sub_component, str):
        return v

    # resolve data_config_alias to data config
    components = values.get("data_configs") or []
    component_map = {cmp.name: cmp for cmp in components}

    # resolve sub component name to component config
    if sub_component not in component_map:
        raise ValueError(f"{data_config_alias} {sub_component} not found in {components}")

    v[data_config_alias] = component_map[sub_component]
    return v


def _resolve_evaluator(v, values):
    if not isinstance(v, dict):
        return v

    evaluator = v.get("evaluator")
    if isinstance(evaluator, dict):
        for idx, metric in enumerate(evaluator.get("metrics", [])):
            evaluator["metrics"][idx] = _resolve_data_config(metric, values, "data_config")
        return v
    elif not isinstance(evaluator, str):
        return v

    # resolve evaluator name to evaluators member config
    if "evaluators" not in values:
        raise ValueError("Invalid evaluators")
    evaluators = values["evaluators"] or {}
    if evaluator not in evaluators:
        raise ValueError(f"Evaluator {evaluator} not found in evaluators")
    v["evaluator"] = evaluators[evaluator]
    return v


def _have_aml_client(config_item, values):
    resource_path_type = config_item.get("type")
    if resource_path_type in AZUREML_RESOURCE_TYPES:
        rp_aml_client = config_item.get("config", {}).get("azureml_client")
        if not rp_aml_client:
            if "azureml_client" not in values:
                raise ValueError(f"azureml_client is required for azureml resource path in config of {config_item}")
            return True
    return False
