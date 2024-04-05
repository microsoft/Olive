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
from olive.common.pydantic_v1 import validator
from olive.data.config import DataConfig
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine, EngineConfig
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes import FullPassConfig
from olive.passes.pass_config import PassParamDefault
from olive.resource_path import AZUREML_RESOURCE_TYPES
from olive.systems.system_config import SystemConfig

logger = logging.getLogger(__name__)


class RunPassConfig(FullPassConfig):
    host: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    clean_run_cache: bool = False
    output_name: str = None


class RunEngineConfig(EngineConfig):
    evaluate_input_model: bool = True
    output_dir: Union[Path, str] = None
    output_name: str = None
    packaging_config: Union[PackagingConfig, List[PackagingConfig]] = None
    log_severity_level: int = 1
    ort_log_severity_level: int = 3
    ort_py_log_severity_level: int = 3
    log_to_file: bool = False

    def create_engine(self):
        config = self.dict()
        to_del = [
            "evaluate_input_model",
            "output_dir",
            "output_name",
            "packaging_config",
            "log_severity_level",
            "ort_log_severity_level",
            "ort_py_log_severity_level",
            "log_to_file",
        ]
        for key in to_del:
            del config[key]
        return Engine(config)


INPUT_MODEL_DATA_CONFIG = "__input_model_data_config__"


class RunConfig(ConfigBase):
    azureml_client: AzureMLClientConfig = None
    input_model: ModelConfig
    systems: Dict[str, SystemConfig] = None
    data_root: str = None
    data_configs: Dict[str, DataConfig] = None
    evaluators: Dict[str, OliveEvaluatorConfig] = None
    engine: RunEngineConfig = None
    pass_flows: List[List[str]] = None
    passes: Dict[str, RunPassConfig] = None
    auto_optimizer_config: AutoOptimizerConfig = None

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

    @validator("engine", pre=True, always=True)
    def default_engine_config(cls, v):
        if v is None:
            v = {}
        return v

    @validator("data_configs", pre=True, always=True)
    def insert_input_model_data_config(cls, v, values):
        if "input_model" not in values:
            raise ValueError("Invalid input model")

        if not v:
            # if data_configs is None, create an empty dict
            v = {}

        if INPUT_MODEL_DATA_CONFIG in v:
            raise ValueError(f"Data config name {INPUT_MODEL_DATA_CONFIG} is reserved. Please use another name.")

        # insert input model hf data config if present
        hf_config = values["input_model"].dict()["config"].get("hf_config", {})
        hf_config_dataset = hf_config.get("dataset", None)
        if hf_config_dataset:
            params_config = {
                "model_name": hf_config.get("model_name", None),
                "task": hf_config.get("task", None),
                **hf_config_dataset,
            }
            # insert trust_remote_code from from_pretrained_args if present
            # won't override if value was set to False explicitly
            # will keep as list of keys for future extension
            for key in ["trust_remote_code"]:
                if hf_config.get("from_pretrained_args", {}).get(key) and params_config.get(key) is None:
                    params_config[key] = hf_config["from_pretrained_args"][key]
            v[INPUT_MODEL_DATA_CONFIG] = {
                "name": INPUT_MODEL_DATA_CONFIG,
                "type": HuggingfaceContainer.__name__,
                "params_config": params_config,
            }
        return v

    @validator("data_configs", pre=True)
    def validate_data_config_names(cls, v):
        if not v:
            return v

        # validate data config name is unique
        data_name_set = set()
        for data_config in v.values():
            data_config_obj = validate_config(data_config, DataConfig)
            if data_config_obj.name in data_name_set:
                raise ValueError(f"Data config name {data_config_obj.name} is duplicated. Please use another name.")
            data_name_set.add(data_config_obj.name)
        return v

    @validator("data_configs", pre=True, each_item=True)
    def validate_data_configs(cls, v, values):
        if "input_model" not in values:
            raise ValueError("Invalid input model")

        hf_config = values["input_model"].dict()["config"].get("hf_config", {})

        if isinstance(v, DataConfig):
            v = v.dict()

        if v["name"] == INPUT_MODEL_DATA_CONFIG:
            # skip validation for input model data config
            return v

        if v["type"] == HuggingfaceContainer.__name__:
            # auto insert model_name and task from input model hf config if not present
            # both are required for huggingface container
            for key in ["model_name", "task"]:
                if not v["params_config"].get(key, None):
                    v["params_config"][key] = hf_config.get(key, None)
            # auto insert trust_remote_code from input model hf config
            # won't override if value was set to False explicitly
            for key in ["trust_remote_code"]:
                if (
                    hf_config.get("from_pretrained_args", {}).get(key, None)
                    and v["params_config"].get(key, None) is None
                ):
                    v["params_config"][key] = hf_config["from_pretrained_args"][key]

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
                # we won't auto insert the input model data config for pass
                # user must explicitly set the data config to INPUT_MODEL_DATA_CONFIG if needed
                v["config"] = _resolve_data_config(v["config"], values, param_name, auto_insert=False)

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

    @validator("auto_optimizer_config", always=True)
    def validate_auto_optimizer_config(cls, v, values):
        if not v:
            v = AutoOptimizerConfig()
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
    if system_alias in v:
        v[system_alias] = validate_config(v[system_alias], SystemConfig)
        if v[system_alias].type == "AzureML":
            if not values["azureml_client"]:
                raise ValueError("AzureML client config is required for AzureML system")
            v[system_alias].config.azureml_client_config = values["azureml_client"]
    return v


def _resolve_data_config(v, values, data_config_alias, auto_insert=True):
    # get the value for data_config_alias in v
    data_container_config = v.get(data_config_alias, None)
    # auto insert input model data config if data_container_config is None
    if not data_container_config and INPUT_MODEL_DATA_CONFIG in values["data_configs"] and auto_insert:
        v[data_config_alias] = INPUT_MODEL_DATA_CONFIG
    # resolve data_config_alias to data config
    return _resolve_config_str(v, values, data_config_alias, component_name="data_configs")


def _resolve_evaluator(v, values):
    if not isinstance(v, dict):
        return v

    evaluator = v.get("evaluator")
    if isinstance(evaluator, dict):
        for idx, metric in enumerate(evaluator.get("metrics", [])):
            evaluator["metrics"][idx] = _resolve_data_config(metric, values, "data_config")
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
                raise ValueError(f"azureml_client is required for azureml resource path in config if {config_item}")
            return True
    return False
