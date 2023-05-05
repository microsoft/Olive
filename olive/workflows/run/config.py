# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase, validate_config
from olive.data_container.config import DataContainerConfig
from olive.data_container.constants import DEFAULT_HF_DATA_CONTAINER_NAME, DefaultDataContainer
from olive.data_container.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine, EngineConfig
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes import FullPassConfig
from olive.systems.system_config import SystemConfig


class RunPassConfig(FullPassConfig):
    host: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    clean_run_cache: bool = False


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
    verbose: bool = False
    input_model: ModelConfig
    systems: Dict[str, SystemConfig] = None
    data_container: Dict[str, DataContainerConfig] = {
        DefaultDataContainer.DATA_CONTAINER.value: DataContainerConfig(),
        DEFAULT_HF_DATA_CONTAINER_NAME: DataContainerConfig(
            name=DEFAULT_HF_DATA_CONTAINER_NAME,
            type=HuggingfaceContainer.__name__,
        )
    }
    evaluators: Dict[str, OliveEvaluatorConfig] = None
    engine: RunEngineConfig
    passes: Dict[str, RunPassConfig]


    @validator("data_container", pre=True, each_item=True, always=True)
    def validate_data_container(cls, v, values):
        if isinstance(v, DataContainerConfig):
            v = v.dict()

        hf_config = values["input_model"].dict()["config"].get("hf_config", {})
        if v["type"] == HuggingfaceContainer.__name__:
            if hf_config:
                v["params_config"]["model_name"] = hf_config["model_name"]
                v["params_config"]["task_type"] = hf_config["task"]
                v["params_config"].update(hf_config.get("dataset", {}))
        return validate_config(v, DataContainerConfig)

    @validator("evaluators", pre=True, each_item=True)
    def validate_evaluators(cls, v, values):
        v = _resolve_system(v, values, "target")
        for idx, metric in enumerate(v.get("metrics", [])):
            v["metrics"][idx] = _resolve_data_container(metric, values, "data_container")
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
        if v["type"] == "OnnxQuantization" or v["type"] == "OnnxStaticQuantization":
            v["config"] = _resolve_data_container(v.get("config", {}), values, "data_container")
        return v


def _resolve_config_str(v, values, alias, component_name="systems"):
    if not isinstance(v, dict):
        return v

    sub_component = v.get(alias)
    if not isinstance(sub_component, str):
        return v

    # resolve system name to systems member config
    if component_name not in values:
        raise ValueError("Invalid systems")
    components = values[component_name] or {}
    if sub_component not in components:
        raise ValueError(f"{alias} {sub_component} not found in systems")
    v[alias] = components[sub_component]
    return v


def _resolve_system(v, values, system_alias, component_name="systems"):
    return _resolve_config_str(v, values, system_alias, component_name=component_name)


def _resolve_data_container(v, values, system_alias, component_name="data_container"):
    data_container_config = v.get("data_container", None)
    hf_data_config = values["input_model"].dict()["config"].get("hf_config", {}).get("dataset", None)
    if not data_container_config and hf_data_config:
        # if data_container is None, we need to update the config to use HuggingfaceContainer
        v["data_container"] = DEFAULT_HF_DATA_CONTAINER_NAME
    return _resolve_config_str(v, values, system_alias, component_name=component_name)


def _resolve_evaluator(v, values):
    if not isinstance(v, dict):
        return v

    evaluator = v.get("evaluator")
    if isinstance(evaluator, dict):
        v["evaluator"] = _resolve_system(evaluator, values, "target")
        for metrics in v["evaluator"].get("metrics", []):
            metrics = _resolve_data_container(metrics, values, "data_container")
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
