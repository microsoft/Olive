# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Union

from pydantic import validator

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, validate_config
from olive.data.config import DataConfig
from olive.data.constants import DEFAULT_HF_DATA_CONTAINER_NAME, DefaultDataContainer
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine, EngineConfig
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes import FullPassConfig, Pass
from olive.resource_path import AZUREML_RESOURCE_TYPES
from olive.systems.system_config import SystemConfig


class RunPassConfig(FullPassConfig):
    host: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    clean_run_cache: bool = False
    output_name: str = None


class RunEngineConfig(EngineConfig):
    evaluation_only: bool = False
    output_dir: Union[Path, str] = None
    output_name: str = None
    packaging_config: PackagingConfig = None
    log_severity_level: int = 1
    ort_log_severity_level: int = 3

    def create_engine(self):
        config = self.dict()
        to_del = [
            "evaluation_only",
            "output_dir",
            "output_name",
            "packaging_config",
            "log_severity_level",
            "ort_log_severity_level",
        ]
        for key in to_del:
            del config[key]
        return Engine(config)


class RunConfig(ConfigBase):
    azureml_client: AzureMLClientConfig = None
    input_model: ModelConfig
    systems: Dict[str, SystemConfig] = None
    data_configs: Dict[str, DataConfig] = {
        DefaultDataContainer.DATA_CONTAINER.value: DataConfig(),
        DEFAULT_HF_DATA_CONTAINER_NAME: DataConfig(
            name=DEFAULT_HF_DATA_CONTAINER_NAME,
            type=HuggingfaceContainer.__name__,
        ),
    }
    evaluators: Dict[str, OliveEvaluatorConfig] = None
    engine: RunEngineConfig
    passes: Dict[str, RunPassConfig]

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

    @validator("data_configs", pre=True, each_item=True, always=True)
    def validate_data_configs(cls, v, values):
        if "input_model" not in values:
            raise ValueError("Invalid input model")

        hf_config = values["input_model"].dict()["config"].get("hf_config", {})

        if isinstance(v, DataConfig):
            # clean up default components before config validation
            v.components = None if hf_config.get("dataset", None) else v.components
            v = v.dict()

        if v["type"] == HuggingfaceContainer.__name__:
            if hf_config:
                v["params_config"]["model_name"] = hf_config.get("model_name", None)
                v["params_config"]["task"] = hf_config.get("task", None)
                v["params_config"].update(hf_config.get("dataset", {}))
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
    def validate_evaluation_only(cls, v):
        if v.evaluation_only and v.evaluator is None:
            raise ValueError("Evaluation only requires evaluator")
        return v

    @validator("passes", pre=True, each_item=True)
    def validate_pass_host_evaluator(cls, v, values):
        v = _resolve_system(v, values, "host")
        return _resolve_evaluator(v, values)

    @validator("passes", pre=True, each_item=True)
    def validate_pass_search(cls, v, values):
        if "engine" not in values:
            raise ValueError("Invalid engine")

        if not values["engine"].search_strategy:
            # disable search if search_strategy is None/False/{}, user cannot override
            disable_search = True
        else:
            # disable search if user explicitly set disable_search to True
            disable_search = v.get("disable_search", False)

        v["disable_search"] = disable_search
        pass_cls = Pass.registry.get(v["type"].lower(), None)
        if pass_cls and pass_cls.requires_data_config():
            v["config"] = _resolve_data_config(v.get("config", {}), values, "data_config")

            if not v["config"]:
                return v
            data_dir_config = v["config"].get("data_dir", None)
            if isinstance(data_dir_config, dict):
                if _have_aml_client(data_dir_config, values):
                    data_dir_config["config"]["azureml_client"] = values["azureml_client"]
                v["config"]["data_dir"] = data_dir_config

        if pass_cls and pass_cls.requires_eval_config():
            v["config"] = _resolve_eval_config(v.get("config", {}), values, "evaluator")
            if not v["config"]:
                return v
        return v


def _resolve_config_str(v, values, alias, component_name):
    if not isinstance(v, dict):
        return v

    sub_component = v.get(alias)
    if not isinstance(sub_component, str):
        return v

    # resolve component name to component config
    if component_name not in values:
        raise ValueError(f"Invalid {component_name}")
    components = values[component_name] or {}
    if sub_component not in components:
        raise ValueError(f"{alias} {sub_component} not found in {components}")
    v[alias] = components[sub_component]
    return v


def _resolve_system(v, values, system_alias, component_name="systems"):
    v = _resolve_config_str(v, values, system_alias, component_name=component_name)
    if system_alias in v:
        v[system_alias] = validate_config(v[system_alias], SystemConfig)
        if v[system_alias].type == "AzureML":
            if not values["azureml_client"]:
                raise ValueError("AzureML client config is required for AzureML system")
            v[system_alias].config.azureml_client_config = values["azureml_client"]
    return v


def _resolve_data_config(v, values, data_config_alias, component_name="data_configs"):
    data_container_config = v.get("data_config", None)
    if "input_model" not in values:
        raise ValueError("Invalid input model")
    hf_data_config = values["input_model"].dict()["config"].get("hf_config", {}).get("dataset", None)
    if not data_container_config and hf_data_config:
        # if data_container is None, we need to update the config to use HuggingfaceContainer
        v["data_config"] = DEFAULT_HF_DATA_CONTAINER_NAME
    return _resolve_config_str(v, values, data_config_alias, component_name=component_name)


def _resolve_eval_config(v, values, eval_config_alias, component_name="evaluators"):
    if "input_model" not in values:
        raise ValueError("Invalid input model")
    return _resolve_config_str(v, values, eval_config_alias, component_name=component_name)


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
