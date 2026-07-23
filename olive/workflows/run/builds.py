# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import ConfigDict, Field, StringConstraints

from olive.common.config_utils import ConfigBase, load_config_file
from olive.common.constants import DEFAULT_WORKFLOW_ID
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.search.search_strategy import SearchStrategyConfig
from olive.systems.common import SystemType
from olive.systems.system_config import SystemConfig
from olive.workflows.run.config import RunConfig

BUILD_DEFAULT_KEY = "_default"
BUILD_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class BuildConfigPartial(ConfigBase):
    """Partial build configuration used by the ``_default`` entry."""

    model_config = ConfigDict(extra="forbid")

    components: Optional[list[NonEmptyString]] = Field(None, min_length=1)
    pipeline: Optional[list[NonEmptyString]] = Field(None, min_length=1)
    output_dir: Optional[NonEmptyString] = None
    host: Optional[Union[SystemConfig, str]] = None
    target: Optional[Union[SystemConfig, str]] = None
    evaluator: Optional[Union[OliveEvaluatorConfig, str]] = None
    search_strategy: Optional[Union[SearchStrategyConfig, bool]] = None


class BuildConfig(BuildConfigPartial):
    """A build that expands into one ordinary Olive run configuration."""

    pipeline: list[NonEmptyString] = Field(..., min_length=1)
    output_dir: NonEmptyString = Field(...)


def parse_run_config(
    run_config: Union[str, Path, dict],
) -> Union[RunConfig, OrderedDict[str, RunConfig]]:
    """Parse one ordinary run config or expand and prevalidate every configured build."""
    raw_run_config = deepcopy(run_config) if isinstance(run_config, dict) else load_config_file(run_config)
    if not isinstance(raw_run_config, dict):
        raise TypeError("Olive run configuration must be a dictionary.")
    if "builds" not in raw_run_config:
        return RunConfig.model_validate(raw_run_config)

    parsed_builds = OrderedDict()
    for build_name, build_config in expand_builds(raw_run_config).items():
        try:
            parsed_build = RunConfig.model_validate(deepcopy(build_config))
            _validate_build_host(parsed_build)
            parsed_builds[build_name] = parsed_build
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid build {build_name!r}: {exc}") from exc
    return parsed_builds


def _validate_build_host(run_config: RunConfig) -> None:
    host = run_config.engine.host
    if host is not None and host.type != SystemType.Local:
        raise ValueError(f"Multi-build workflows currently support only LocalSystem hosts; got {host.type.value!r}.")


def expand_builds(run_config: dict) -> OrderedDict[str, dict]:
    """Expand ``builds`` into independent, ordinary Olive run configurations."""
    if not isinstance(run_config, dict):
        raise TypeError("Multi-build configuration must be a dictionary.")

    source_config = deepcopy(run_config)
    if "builds" not in source_config:
        return OrderedDict()
    raw_builds = source_config.pop("builds")
    if not isinstance(raw_builds, dict):
        raise ValueError("`builds` must be a dictionary keyed by build name.")

    builds = _parse_builds(raw_builds)
    passes = source_config.get("passes") or {}
    workflow_id = source_config.get("workflow_id", DEFAULT_WORKFLOW_ID)
    expanded = OrderedDict()

    for build_name, build in builds.items():
        missing_passes = [pass_name for pass_name in build.pipeline if pass_name not in passes]
        if missing_passes:
            raise ValueError(
                f"Build {build_name!r} references unknown pass(es) {missing_passes}. Known passes: {sorted(passes)}."
            )

        child_config = deepcopy(source_config)
        child_config["workflow_id"] = f"{workflow_id}_{build_name}"
        child_config["passes"] = OrderedDict((pass_name, deepcopy(passes[pass_name])) for pass_name in build.pipeline)
        _set_engine_value(child_config, "output_dir", build.output_dir)

        for field_name in ("host", "target", "evaluator", "search_strategy"):
            value = getattr(build, field_name)
            if value is not None:
                _set_engine_value(child_config, field_name, value)

        if build.components:
            input_model = child_config.get("input_model")
            if input_model is None:
                raise ValueError(f"Build {build_name!r} selects components but no input_model is configured.")
            child_config["input_model"] = (
                ModelConfig.model_validate(deepcopy(input_model)).select_components(build.components).model_dump()
            )

        expanded[build_name] = child_config

    return expanded


def _parse_builds(raw_builds: dict) -> OrderedDict[str, BuildConfig]:
    default_raw = raw_builds.get(BUILD_DEFAULT_KEY, {})
    if not isinstance(default_raw, dict):
        raise ValueError("`builds._default` must be a dictionary.")
    default_config = BuildConfigPartial.model_validate(default_raw).model_dump(exclude_none=True)
    builds = OrderedDict()

    for build_name, raw_build in raw_builds.items():
        if build_name == BUILD_DEFAULT_KEY:
            continue
        if not isinstance(build_name, str) or not BUILD_NAME_PATTERN.fullmatch(build_name):
            raise ValueError(f"Invalid build name {build_name!r}. Use letters, numbers, dots, underscores, or hyphens.")
        if not isinstance(raw_build, dict):
            raise ValueError(f"Build {build_name!r} must be a dictionary.")
        builds[build_name] = BuildConfig.model_validate({**default_config, **raw_build})

    if not builds:
        raise ValueError("`builds` must contain at least one named build in addition to `_default`.")
    return builds


def _set_engine_value(run_config: dict, field_name: str, value) -> None:
    run_config.pop(field_name, None)
    engine_config = run_config.get("engine") or {}
    if hasattr(engine_config, "model_dump"):
        engine_config = engine_config.model_dump()
    elif not isinstance(engine_config, dict):
        raise ValueError("`engine` must be a dictionary.")
    else:
        engine_config = deepcopy(engine_config)

    engine_config[field_name] = value.model_dump() if hasattr(value, "model_dump") else deepcopy(value)
    run_config["engine"] = engine_config
