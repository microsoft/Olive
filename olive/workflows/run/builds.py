# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from typing import Optional, Union

from olive.cache import CacheConfig
from olive.common.config_utils import load_config_file
from olive.common.constants import DEFAULT_WORKFLOW_ID
from olive.model import ModelConfig
from olive.systems.common import SystemType
from olive.workflows.run.config import BuildConfig, BuildConfigPartial, RunConfig

BUILD_DEFAULT_KEY = "_default"
BUILD_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DEFAULT_MAX_CONCURRENT_BUILDS = None
MAX_CONCURRENT_BUILDS_KEY = "max_concurrent_builds"


def get_build_output_dir(
    build_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    default_output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Return an explicit build output directory or a build-specific directory under the configured parent."""
    if output_dir:
        return output_dir
    return str(Path(default_output_dir or "output") / build_name)


class MultiBuildRunConfig(OrderedDict[str, RunConfig]):
    """Parsed build configurations and their execution-level concurrency limit."""

    def __init__(
        self,
        *args,
        max_concurrent_builds: Optional[int] = DEFAULT_MAX_CONCURRENT_BUILDS,
        output_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_concurrent_builds = max_concurrent_builds or max(len(self), 1)
        self.output_dir = output_dir


def parse_run_config(
    run_config: Union[str, Path, dict],
) -> Union[RunConfig, MultiBuildRunConfig]:
    """Parse one ordinary run config or expand and prevalidate every configured build.

    Multi-build execution is parallel by default. Set the top-level
    ``max_concurrent_builds`` field to bound concurrency or force serial execution with 1.
    """
    raw_run_config = deepcopy(run_config) if isinstance(run_config, dict) else load_config_file(run_config)
    if not isinstance(raw_run_config, dict):
        raise TypeError("Olive run configuration must be a dictionary.")
    if raw_run_config.get("builds") is None:
        return RunConfig.model_validate(raw_run_config)

    max_concurrent_builds = _parse_max_concurrent_builds(raw_run_config)
    raw_run_config.pop(MAX_CONCURRENT_BUILDS_KEY, None)
    output_dir = _get_workflow_output_dir(raw_run_config)
    parsed_builds = OrderedDict()
    for build_name, build_config in expand_builds(raw_run_config).items():
        try:
            parsed_build = RunConfig.model_validate(deepcopy(build_config))
            _validate_build_host(parsed_build)
            parsed_builds[build_name] = parsed_build
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid build {build_name!r}: {exc}") from exc
    _validate_build_write_dirs(parsed_builds)
    return MultiBuildRunConfig(
        parsed_builds,
        max_concurrent_builds=max_concurrent_builds,
        output_dir=output_dir,
    )


def _get_workflow_output_dir(run_config: dict) -> Optional[Path]:
    output_dir = run_config.get("output_dir")
    if output_dir is None:
        engine = run_config.get("engine") or {}
        if hasattr(engine, "model_dump"):
            engine = engine.model_dump()
        if not isinstance(engine, dict):
            raise ValueError("`engine` must be a dictionary.")
        output_dir = engine.get("output_dir")
    return Path(output_dir).resolve() if output_dir is not None else None


def _parse_max_concurrent_builds(run_config: dict) -> Optional[int]:
    value = run_config.get(MAX_CONCURRENT_BUILDS_KEY, DEFAULT_MAX_CONCURRENT_BUILDS)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`{MAX_CONCURRENT_BUILDS_KEY}` must be a positive integer; got {value!r}.")
    return value


def _validate_build_host(run_config: RunConfig) -> None:
    host = run_config.engine.host
    if host is not None and host.type != SystemType.Local:
        raise ValueError(f"Multi-build workflows currently support only LocalSystem hosts; got {host.type.value!r}.")


def _validate_build_write_dirs(build_configs: dict[str, RunConfig]) -> None:
    write_dirs = {name: _get_build_write_dirs(config) for name, config in build_configs.items()}
    for (first_name, first_dirs), (second_name, second_dirs) in combinations(write_dirs.items(), 2):
        for (first_type, first_dir), (second_type, second_dir) in product(first_dirs.items(), second_dirs.items()):
            if _paths_overlap(first_dir, second_dir):
                raise ValueError(
                    f"Parallel builds {first_name!r} and {second_name!r} have overlapping writable directories: "
                    f"{first_type} directory {first_dir} and {second_type} directory {second_dir}."
                )


def _get_build_write_dirs(run_config: RunConfig) -> dict[str, Path]:
    output_dir = Path(run_config.engine.output_dir)
    return {"artifact": output_dir.resolve(), "cache": get_build_cache_dir(run_config)}


def get_build_cache_dir(run_config: RunConfig) -> Path:
    cache_config = run_config.engine.cache_config
    if cache_config is None:
        cache_config = CacheConfig(cache_dir=run_config.engine.cache_dir)
    elif isinstance(cache_config, dict):
        cache_config = CacheConfig.model_validate(cache_config)
    return (Path(cache_config.get_local_cache_dir()) / run_config.workflow_id).resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def expand_builds(run_config: dict) -> OrderedDict[str, dict]:
    """Expand ``builds`` into independent, ordinary Olive run configurations."""
    if not isinstance(run_config, dict):
        raise TypeError("Multi-build configuration must be a dictionary.")

    source_config = deepcopy(run_config)
    _parse_max_concurrent_builds(source_config)
    source_config.pop(MAX_CONCURRENT_BUILDS_KEY, None)
    if "builds" not in source_config:
        return OrderedDict()
    raw_builds = source_config.pop("builds")
    if not isinstance(raw_builds, dict):
        raise ValueError("`builds` must be a dictionary keyed by build name.")

    builds = _parse_builds(raw_builds, _get_workflow_output_dir(source_config))
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
        _set_engine_value(child_config, "output_dir", get_build_output_dir(build_name, build.output_dir))

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


def _parse_builds(raw_builds: dict, workflow_output_dir: Optional[Path] = None) -> OrderedDict[str, BuildConfig]:
    default_raw = raw_builds.get(BUILD_DEFAULT_KEY, {})
    if not isinstance(default_raw, dict):
        raise ValueError("`builds._default` must be a dictionary.")
    default_config = BuildConfigPartial.model_validate(default_raw).model_dump(exclude_none=True)
    if workflow_output_dir is not None and "output_dir" not in default_config:
        default_config["output_dir"] = str(workflow_output_dir)
    builds = OrderedDict()

    for build_name, raw_build in raw_builds.items():
        if build_name == BUILD_DEFAULT_KEY:
            continue
        if not isinstance(build_name, str) or not BUILD_NAME_PATTERN.fullmatch(build_name):
            raise ValueError(f"Invalid build name {build_name!r}. Use letters, numbers, dots, underscores, or hyphens.")
        if not isinstance(raw_build, dict):
            raise ValueError(f"Build {build_name!r} must be a dictionary.")
        merged_config = {**default_config, **raw_build}
        if "output_dir" not in raw_build:
            merged_config["output_dir"] = get_build_output_dir(
                build_name,
                default_output_dir=default_config.get("output_dir"),
            )
        builds[build_name] = BuildConfig.model_validate(merged_config)

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
