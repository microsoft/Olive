# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, FieldValidationInfo, field_validator
from typing_extensions import Annotated

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, model_dump, validate_config
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
    host: Optional[SystemConfig] = None
    evaluator: Optional[OliveEvaluatorConfig] = None
    clean_run_cache: bool = False
    output_name: Optional[str] = None


class RunEngineConfig(EngineConfig):
    evaluate_input_model: bool = True
    output_dir: Optional[Union[Path, str]] = None
    output_name: Optional[str] = None
    packaging_config: Optional[PackagingConfig] = None
    log_severity_level: int = 1
    ort_log_severity_level: int = 3

    def create_engine(self):
        config = model_dump(self)
        to_del = [
            "evaluate_input_model",
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
    azureml_client: Optional[AzureMLClientConfig] = None
    input_model: ModelConfig
    systems: Optional[Dict[str, SystemConfig]] = None
    data_root: Optional[str] = None
    data_configs: Annotated[Dict[str, DataConfig], Field(validate_default=True)] = {
        DefaultDataContainer.DATA_CONTAINER.value: DataConfig(),
        DEFAULT_HF_DATA_CONTAINER_NAME: DataConfig(
            name=DEFAULT_HF_DATA_CONTAINER_NAME,
            type=HuggingfaceContainer.__name__,
        ),
    }
    evaluators: Optional[Dict[str, OliveEvaluatorConfig]] = None
    pass_flows: Optional[List[List[str]]] = None
    engine: RunEngineConfig
    passes: Dict[str, RunPassConfig]

    @field_validator("input_model", mode="before")
    def insert_aml_client(cls, v, info: FieldValidationInfo):
        if isinstance(v, ModelConfig):
            v = v.model_dump()

        input_model_path = v["config"].get("model_path")
        # if not a dict, return the original value
        if not input_model_path or not isinstance(input_model_path, dict):
            return v

        if _have_aml_client(input_model_path, info.data):
            v["config"]["model_path"]["config"]["azureml_client"] = info.data["azureml_client"]
        return v

    @field_validator("data_configs", mode="before")
    def validate_data_configs(cls, v, info: FieldValidationInfo):
        if "input_model" not in info.data:
            raise ValueError("Invalid input model")

        hf_config = info.data["input_model"].config.get("hf_config", {})
        for config_name, data_config in v.items():
            if isinstance(data_config, DataConfig):
                # clean up default components before config validation
                data_config.components = None if hf_config.get("dataset", None) else data_config.components
                data_config = data_config.model_dump()

                if data_config["type"] == HuggingfaceContainer.__name__:
                    if hf_config:
                        data_config["params_config"]["model_name"] = hf_config.get("model_name", None)
                        data_config["params_config"]["task"] = hf_config.get("task", None)
                        data_config["params_config"].update(hf_config.get("dataset", {}))
                v[config_name] = validate_config(data_config, DataConfig)

        return v

    @field_validator("evaluators", mode="before")
    def validate_evaluators(cls, v, info: FieldValidationInfo):
        for evaluator_name, evaluator in v.items():
            for idx, metric in enumerate(evaluator.get("metrics", [])):
                metric = _resolve_data_config(metric, info.data, "data_config")

                data_dir_config = metric.get("user_config", {}).get("data_dir", None)
                if isinstance(data_dir_config, dict):
                    if _have_aml_client(data_dir_config, info.data):
                        data_dir_config["config"]["azureml_client"] = info.data["azureml_client"]
                    if "user_config" not in metric:
                        metric["user_config"] = {}
                    metric["user_config"]["data_dir"] = data_dir_config
        return v

    @field_validator("engine", mode="before")
    def validate_engine(cls, v, info: FieldValidationInfo):
        v = _resolve_system(v, info.data, "host")
        v = _resolve_system(v, info.data, "target")
        return _resolve_evaluator(v, info.data)

    @field_validator("engine")
    def validate_evaluate_input_model(cls, v):
        if v.evaluate_input_model and v.evaluator is None:
            raise ValueError("Evaluation only requires evaluator")
        return v

    @field_validator("passes", mode="before")
    def validate_pass_search(cls, v, info: FieldValidationInfo):
        if "engine" not in info.data:
            raise ValueError("Invalid engine")

        for pass_name, pass_config in v.items():
            if not info.data["engine"].search_strategy:
                # disable search if search_strategy is None/False/{}, user cannot override
                disable_search = True
            else:
                # disable search if user explicitly set disable_search to True
                disable_search = pass_config.get("disable_search", False)

            pass_config["disable_search"] = disable_search

            pass_cls = Pass.registry.get(pass_config["type"].lower(), None)
            if pass_cls and pass_cls.requires_data_config():
                pass_config["config"] = _resolve_data_config(pass_config.get("config", {}), info.data, "data_config")
                if pass_config["config"]:
                    data_dir_config = pass_config["config"].get("data_dir", None)
                    if isinstance(data_dir_config, dict):
                        if _have_aml_client(data_dir_config, info.data):
                            data_dir_config["config"]["azureml_client"] = info.data["azureml_client"]
                        pass_config["config"]["data_dir"] = data_dir_config

            pass_config = _resolve_system(pass_config, info.data, "host")
            pass_config = _resolve_evaluator(pass_config, info.data)
            v[pass_name] = pass_config

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
    hf_data_config = values["input_model"].model_dump()["config"].get("hf_config", {}).get("dataset", None)
    if not data_container_config and hf_data_config:
        # if data_container is None, we need to update the config to use HuggingfaceContainer
        v["data_config"] = DEFAULT_HF_DATA_CONTAINER_NAME
    return _resolve_config_str(v, values, data_config_alias, component_name=component_name)


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
