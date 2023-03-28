# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase
from olive.engine import EngineConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes import FullPassConfig
from olive.systems.system_config import SystemConfig


class RunPassConfig(FullPassConfig):
    host: SystemConfig = None
    evaluator: OliveEvaluatorConfig = None
    clean_run_cache: bool = False


class RunEngineConfig(EngineConfig):
    output_dir: Union[Path, str] = None
    output_name: str = None


class RunConfig(ConfigBase):
    verbose: bool = False
    input_model: ModelConfig
    systems: Dict[str, SystemConfig] = None
    evaluators: Dict[str, OliveEvaluatorConfig] = None
    engine: RunEngineConfig
    passes: Dict[str, RunPassConfig]

    @validator("evaluators", pre=True, each_item=True)
    def validate_evaluators(cls, v, values):
        return _resolve_system(v, values, "target")

    @validator("engine", pre=True)
    def validate_engine(cls, v, values):
        v = _resolve_system(v, values, "host")
        return _resolve_evaluator(v, values)

    @validator("passes", pre=True, each_item=True)
    def validate_pass_host(cls, v, values):
        v = _resolve_system(v, values, "host")
        return _resolve_evaluator(v, values)


def _resolve_system(v, values, system_alias):
    if not isinstance(v, dict):
        return v

    system = v.get(system_alias)
    if not isinstance(system, str):
        return v

    # resolve system name to systems member config
    if "systems" not in values:
        raise ValueError("Invalid systems")
    systems = values["systems"] or {}
    if system not in systems:
        raise ValueError(f"{system_alias} {system} not found in systems")
    v[system_alias] = systems[system]
    return v


def _resolve_evaluator(v, values):
    if not isinstance(v, dict):
        return v

    evaluator = v.get("evaluator")
    if isinstance(evaluator, dict):
        v["evaluator"] = _resolve_system(evaluator, values, "target")
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
