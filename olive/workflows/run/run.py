# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import importlib
import json
import logging
import os
from copy import deepcopy
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Optional, Union

from olive.common.config_utils import load_config_file
from olive.common.utils import hash_dict, set_tempdir
from olive.hardware.constants import ExecutionProvider
from olive.logging import set_default_logger_severity, set_ort_logger_severity, set_verbosity_info
from olive.package_config import OlivePackageConfig
from olive.systems.accelerator_creator import create_accelerator
from olive.systems.common import SystemType
from olive.telemetry.constants import SUPPRESS_WORKFLOW_TELEMETRY_ENV
from olive.telemetry.telemetry import is_ci_environment
from olive.telemetry.telemetry_extensions import _format_exception_message, log_error, log_recipe_result
from olive.workflows.run.config import RunConfig

if TYPE_CHECKING:
    from olive.engine.config import RunPassConfig

logger = logging.getLogger(__name__)
RECIPE_HASH_REDACTED_VALUE = "<resource>"
CONFIG_REFERENCE_REDACTED_VALUE = "<reference>"
CONFIG_CALLABLE_REDACTED_VALUE = "<callable>"
RECIPE_HASH_REDACTED_KEYS = {
    "output_dir",
    "cache_dir",
    "tempdir",
    "additional_files",
    "dockerfile",
    "build_context_path",
    "python_environment_path",
    "prepend_to_path",
    "script_dir",
    "model_script",
    # package_config is tracked separately via package_config_provided and
    # package_config_overrides, but excluded from recipe_hash because it is an
    # environment/infrastructure path.
    "package_config",
    "work_dir",
}
CONFIG_SNAPSHOT_REDACTED_KEYS = RECIPE_HASH_REDACTED_KEYS | {
    "model_path",
    "_name_or_path",
    "adapter_path",
    "user_script",
}
CONFIG_REFERENCE_KEYS = {"host", "target", "evaluator"}
_NO_OVERRIDE = object()


def get_required_packages(package_config: OlivePackageConfig, run_config: RunConfig) -> set[str]:
    def get_system_extras(host_type, accelerators, execution_providers):
        extra_name = None
        if host_type is None:
            extra_name = "cpu"
        elif host_type == SystemType.Docker:
            extra_name = "docker"
        elif host_type == SystemType.Local:
            if accelerators and "GPU" in list(map(str.upper, accelerators)):
                if execution_providers and ExecutionProvider.DmlExecutionProvider in execution_providers:
                    extra_name = "directml"
                else:
                    extra_name = "gpu"
            else:
                extra_name = "cpu"

        return extra_name

    def get_pass_extras(pass_type):
        pass_module_config = package_config.get_pass_module_config(pass_type)

        extra_results = []
        extra_results.extend(pass_module_config.module_dependencies)
        for extra_name in pass_module_config.extra_dependencies:
            extra_results.extend(package_config.extra_dependencies.get(extra_name, []))
        return extra_results

    extras = deepcopy(package_config.extra_dependencies)
    ort_packages = extras.get("ort", [])

    # add dependencies for passes
    required_packages = []
    if run_config.passes:
        for passes_configs in run_config.passes.values():
            for pass_config in passes_configs:
                host = pass_config.host or run_config.engine.host
                if (host and host.type == SystemType.Local) or not host:
                    required_packages.extend(get_pass_extras(pass_config.type))

    # add dependencies for engine
    host_type = None
    accelerators = []
    execution_providers = []
    if run_config.engine.host:
        host_type = run_config.engine.host.type
        if run_config.engine.host.config.accelerators:
            for acc in run_config.engine.host.config.accelerators:
                accelerators.append(acc.device)
                if acc.get_ep_strs():
                    execution_providers.extend(acc.get_ep_strs())

    system_extra_name = get_system_extras(host_type, accelerators, execution_providers)
    if system_extra_name:
        required_packages.extend(extras.get(system_extra_name))

    logger.info("The following packages are required in the local environment: %s", required_packages)
    return set.union(set(required_packages), set(ort_packages))


def is_execution_provider_required(run_config: RunConfig, package_config: OlivePackageConfig) -> bool:
    # input model is onnx and we want to evaluate the input model
    # there are passes that produce onnx models
    return (
        run_config.engine.evaluator
        and run_config.engine.evaluate_input_model
        and run_config.input_model.type.lower() == "onnxmodel"
    ) or any(
        package_config.is_onnx_module(p.type)
        for passes_configs in (run_config.passes or {}).values()
        for p in passes_configs
    )


def run_engine(package_config: OlivePackageConfig, run_config: RunConfig):
    workflow_id = run_config.workflow_id
    logger.info("Running workflow %s", workflow_id)

    # for onnxruntime
    # ort_py_log_severity_level: python logging levels
    set_ort_logger_severity(run_config.engine.ort_py_log_severity_level)

    # ort_log_severity_level: C++ logging levels
    try:
        ort = importlib.import_module("onnxruntime")

        ort.set_default_logger_severity(run_config.engine.ort_log_severity_level)
    except Exception:
        logger.warning("ORT log severity level configuration ignored since the module isn't installed.")

    olive_config = run_config.to_json()
    engine = run_config.engine.create_engine(package_config, workflow_id)
    engine.cache.cache_olive_config(olive_config)

    # check if target is not used
    used_passes_configs = get_used_passes_configs(run_config)
    target_not_used = (
        # no evaluator given (also implies no search)
        engine.evaluator_config is None
        # no pass specific evaluator
        # no pass needs to run on target
        and all(
            pass_config.evaluator is None and not get_run_on_target(package_config, pass_config)
            for pass_config in used_passes_configs
        )
    )

    is_ep_required = is_execution_provider_required(run_config, package_config)
    accelerator_spec = create_accelerator(
        engine.target_config, skip_supported_eps_check=target_not_used, is_ep_required=is_ep_required
    )

    # Set passes with the engine
    engine.set_input_passes_configs(run_config.passes)

    # run
    return engine.run(
        run_config.input_model,
        accelerator_spec,
        run_config.engine.packaging_config,
        run_config.engine.output_dir,
        run_config.engine.evaluate_input_model,
        run_config.engine.log_to_file,
        run_config.engine.log_severity_level,
    )


def run(
    run_config: Union[str, Path, dict],
    list_required_packages: bool = False,
    package_config: Optional[Union[str, Path, dict]] = None,
    tempdir: Optional[Union[str, Path]] = None,
    recipe_telemetry_metadata: Optional[dict[str, Any]] = None,
):
    # set tempdir
    set_tempdir(tempdir)

    try:
        run_config_telemetry_input = _load_config_input_for_telemetry(run_config)
    except Exception:
        run_config_telemetry_input = None

    package_config_input = package_config
    try:
        package_config_telemetry_input = (
            _load_config_input_for_telemetry(package_config_input) if package_config_input is not None else None
        )
    except Exception:
        package_config_telemetry_input = None

    package_config_provided = package_config is not None
    if package_config is None:
        package_config = OlivePackageConfig.get_default_config_path()

    parsed_run_config = None
    success = False
    exception = None
    try:
        package_config = OlivePackageConfig.parse_file_or_obj(package_config)
        parsed_run_config = RunConfig.parse_file_or_obj(run_config)

        if list_required_packages:
            # set the log level to INFO for packages
            set_verbosity_info()
            required_packages = get_required_packages(package_config, parsed_run_config)
            generate_files_from_packages(required_packages, "olive_requirements.txt")
            success = True
            return None

        if parsed_run_config.engine.host and parsed_run_config.engine.host.type == SystemType.Docker:
            docker_system = parsed_run_config.engine.host.create_system()
            workflow_output = docker_system.run_workflow(deepcopy(parsed_run_config))
            success = True
            return workflow_output

        # set log level for olive
        set_default_logger_severity(parsed_run_config.engine.log_severity_level)
        workflow_output = run_engine(package_config, parsed_run_config)
        success = True
        return workflow_output
    except Exception as exc:
        exception = exc
        raise
    finally:
        if exception is not None:
            log_error(
                exception_type=type(exception).__name__,
                exception_message=_format_exception_message(exception, exception.__traceback__),
            )
        if os.environ.get(SUPPRESS_WORKFLOW_TELEMETRY_ENV) != "1":
            metadata = _build_recipe_result_metadata(
                run_config,
                run_config_telemetry_input,
                parsed_run_config,
                recipe_telemetry_metadata,
                list_required_packages=list_required_packages,
                package_config_input=package_config_telemetry_input,
                package_config_provided=package_config_provided,
            )
            recipe_name = metadata.pop("recipe_name")
            log_recipe_result(recipe_name, success=success, metadata=metadata)


def generate_files_from_packages(packages, file_name):
    file_path = Path(file_name)
    if file_path.exists():
        logger.warning("%s already exists. Skipping.", file_name)
    else:
        with file_path.open("w") as f:
            f.write("\n".join(packages))
        logger.info("Requirements file %s is generated.", file_name)


def get_used_passes_configs(run_config: RunConfig) -> list["RunPassConfig"]:
    return (
        [pass_config for _, pass_configs in run_config.passes.items() for pass_config in pass_configs]
        if run_config.passes
        else []
    )


def get_run_on_target(package_config: OlivePackageConfig, pass_config: "RunPassConfig") -> bool:
    pass_module_config = package_config.get_pass_module_config(pass_config.type)
    return pass_module_config.run_on_target


def _build_recipe_result_metadata(
    run_config_input: Union[str, Path, dict],
    run_config_telemetry_input: Optional[Any],
    run_config: Optional[RunConfig],
    recipe_telemetry_metadata: Optional[dict[str, Any]],
    *,
    list_required_packages: bool,
    package_config_input: Optional[Union[str, Path, dict]],
    package_config_provided: bool,
) -> dict[str, Any]:
    metadata = dict(recipe_telemetry_metadata or {})
    default_source, default_format = _classify_run_config_source(run_config_input)
    metadata.setdefault("recipe_source", default_source)
    metadata.setdefault("recipe_format", default_format)
    metadata.setdefault("execution_mode", "list_required_packages" if list_required_packages else "run")
    metadata.setdefault("package_config_provided", package_config_provided)
    metadata.setdefault("config_overrides", _build_config_overrides(run_config_telemetry_input))
    if package_config_provided:
        metadata.setdefault("package_config_overrides", _build_package_config_overrides(package_config_input))
    metadata["is_ci"] = is_ci_environment()

    if run_config is None:
        metadata.setdefault("recipe_name", metadata.get("recipe_command") or "WorkflowRun")
        return metadata

    run_config_json = run_config.to_json(make_absolute=False)
    model_metadata = _extract_input_model_metadata(run_config_json["input_model"])
    target_metadata = _extract_target_metadata(run_config)
    host_metadata = _extract_host_metadata(run_config)
    pass_types = [pass_config.type for pass_config in get_used_passes_configs(run_config)]

    metadata.setdefault("recipe_name", metadata.get("recipe_command") or run_config.workflow_id)
    metadata.setdefault("workflow_id", run_config.workflow_id)
    metadata.setdefault("recipe_hash", _build_recipe_hash(run_config_json))
    metadata.setdefault("input_model_type", run_config.input_model.type)
    metadata.setdefault("input_model_source", model_metadata["input_model_source"])
    metadata.setdefault("model_task", model_metadata["model_task"])
    _set_metadata_if_present(metadata, target_metadata)
    _set_metadata_if_present(metadata, host_metadata)
    metadata.setdefault("pass_types", ";".join(pass_types))
    metadata.setdefault("pass_count", len(pass_types))
    metadata.setdefault("data_config_count", len(run_config.data_configs))
    metadata.setdefault("search_enabled", bool(run_config.engine.search_strategy))
    return metadata


def _classify_run_config_source(run_config_input: Any) -> tuple[str, str]:
    if isinstance(run_config_input, dict):
        return "config_dict", "dict"

    if isinstance(run_config_input, (str, PathLike)):
        suffix = Path(run_config_input).suffix.lstrip(".").lower()
        return "config_file", suffix or "unknown"

    return "config_object", "object"


def _build_config_overrides(config_input: Any) -> Optional[str]:
    try:
        config_data = _load_config_input_for_telemetry(config_input)
        if config_data is None:
            return None

        snapshot = _sanitize_config_snapshot(config_data)
        if snapshot in (None, {}, []):
            return None

        return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return None


def _build_package_config_overrides(config_input: Any) -> Optional[str]:
    try:
        config_data = _load_config_input_for_telemetry(config_input)
        if not isinstance(config_data, dict):
            return None

        default_config = _load_default_package_config_for_telemetry()
        baseline = (
            _normalize_package_config_snapshot(default_config) if isinstance(default_config, dict) else _NO_OVERRIDE
        )
        overrides = _extract_config_overrides(_normalize_package_config_snapshot(config_data), baseline)
        if overrides is _NO_OVERRIDE:
            return None

        snapshot = _sanitize_config_snapshot(overrides)
        if not isinstance(snapshot, dict):
            return None

        return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return None


@functools.lru_cache
def _load_default_package_config_for_telemetry() -> Optional[dict[str, Any]]:
    try:
        default_config = load_config_file(OlivePackageConfig.get_default_config_path())
    except Exception:
        return None

    return default_config if isinstance(default_config, dict) else None


def _normalize_package_config_snapshot(config_data: Any) -> Any:
    if not isinstance(config_data, dict):
        return config_data

    normalized = deepcopy(config_data)
    passes = normalized.get("passes")
    if isinstance(passes, dict):
        normalized["passes"] = {str(pass_name).lower(): pass_config for pass_name, pass_config in passes.items()}
    return normalized


def _extract_config_overrides(value: Any, baseline: Any = _NO_OVERRIDE) -> Any:
    if baseline is _NO_OVERRIDE:
        return deepcopy(value)

    if isinstance(value, dict) and isinstance(baseline, dict):
        overrides = {}
        for key, child_value in value.items():
            child_override = _extract_config_overrides(child_value, baseline.get(key, _NO_OVERRIDE))
            if child_override is not _NO_OVERRIDE:
                overrides[key] = child_override
        if overrides:
            return overrides
        return _NO_OVERRIDE if value == baseline else {}

    if isinstance(value, list):
        if isinstance(baseline, list) and value == baseline:
            return _NO_OVERRIDE
        return deepcopy(value)

    if isinstance(value, tuple):
        value_list = list(value)
        baseline_list = list(baseline) if isinstance(baseline, tuple) else baseline
        if isinstance(baseline_list, list) and value_list == baseline_list:
            return _NO_OVERRIDE
        return value_list

    return deepcopy(value) if value != baseline else _NO_OVERRIDE


def _load_config_input_for_telemetry(config_input: Any) -> Optional[Any]:
    if config_input is None:
        return None
    if isinstance(config_input, dict):
        return deepcopy(config_input)
    if isinstance(config_input, (str, PathLike)):
        return load_config_file(config_input)

    model_dump = getattr(config_input, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_defaults=True, exclude_none=True, by_alias=True)
    return None


def _sanitize_config_snapshot(value: Any, key: Optional[str] = None) -> Any:
    if key in CONFIG_SNAPSHOT_REDACTED_KEYS or _is_path_like_key(key):
        return RECIPE_HASH_REDACTED_VALUE
    if key in CONFIG_REFERENCE_KEYS and isinstance(value, str):
        return CONFIG_REFERENCE_REDACTED_VALUE

    if isinstance(value, dict):
        if key == "systems":
            return [_sanitize_config_snapshot(system, "system") for system in value.values()]
        if key == "passes":
            passes = []
            for pass_configs in value.values():
                if isinstance(pass_configs, list):
                    passes.extend(pass_configs)
                else:
                    passes.append(pass_configs)
            return [_sanitize_config_snapshot(pass_config, "pass") for pass_config in passes]
        if key == "evaluators":
            return [_sanitize_config_snapshot(evaluator, "evaluator_config") for evaluator in value.values()]
        return {
            child_key: _sanitize_config_snapshot(child_value, child_key)
            for child_key, child_value in value.items()
            if child_value is not None
        }
    if isinstance(value, list):
        return [_sanitize_config_snapshot(item, key) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_config_snapshot(item, key) for item in value]
    if isinstance(value, Path):
        return RECIPE_HASH_REDACTED_VALUE
    if callable(value):
        return CONFIG_CALLABLE_REDACTED_VALUE
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return f"<{type(value).__name__}>"


def _is_path_like_key(key: Optional[str]) -> bool:
    if key is None:
        return False
    return key in {"path", "paths", "dir", "dirs", "file", "files"} or key.endswith(
        ("_path", "_paths", "_dir", "_dirs", "_file", "_files")
    )


def _extract_input_model_metadata(input_model_config: dict[str, Any]) -> dict[str, Optional[str]]:
    model_config = input_model_config.get("config", {})
    model_attributes = model_config.get("model_attributes", {})
    model_task = model_attributes.get("hf_task") or model_config.get("task")
    raw_identifier = model_attributes.get("_name_or_path") or model_config.get("model_path")
    return {
        "input_model_source": _classify_input_model_source(raw_identifier),
        "model_task": str(model_task) if model_task is not None else None,
    }


def _classify_input_model_source(model_identifier: Any) -> str:
    if model_identifier is None:
        return "unknown"
    if isinstance(model_identifier, dict):
        resource_type = model_identifier.get("type")
        if resource_type == "azureml_registry_model":
            return "azureml"
        return "structured_resource"

    identifier = str(model_identifier)
    if identifier.startswith("azureml://"):
        return "azureml"
    if identifier.startswith("https://huggingface.co/"):
        return "huggingface_url"
    if identifier.startswith(("http://", "https://")):
        return "url"

    if _is_explicit_local_model_path(identifier):
        suffix = PureWindowsPath(identifier).suffix or PurePosixPath(identifier).suffix
        return "local_file" if suffix else "local_folder"
    return "string_name"


def _is_explicit_local_model_path(identifier: str) -> bool:
    return (
        identifier.startswith(("./", "../", ".\\", "..\\", "~/", "~\\", "/", "\\\\"))
        or PureWindowsPath(identifier).is_absolute()
        or PurePosixPath(identifier).is_absolute()
    )


def _extract_target_metadata(run_config: RunConfig) -> dict[str, Optional[str]]:
    target_system = run_config.engine.target
    return _extract_system_metadata(target_system, "target")


def _extract_host_metadata(run_config: RunConfig) -> dict[str, Optional[str]]:
    host_system = run_config.engine.host
    if host_system is None:
        return {
            "host_system_type": SystemType.Local.value,
        }
    return _extract_system_metadata(host_system, "host")


def _extract_system_metadata(system_config: Optional[Any], field_prefix: str) -> dict[str, Optional[str]]:
    system_type = system_config.type.value if system_config is not None else None
    device = None
    execution_provider = None
    execution_providers = None

    accelerators = system_config.config.accelerators if system_config and system_config.config else None
    if accelerators:
        accelerator = accelerators[0]
        device = str(accelerator.device) if accelerator.device is not None else None
        ep_values = accelerator.get_ep_strs() or []
        if ep_values:
            execution_provider = ep_values[0]
            execution_providers = ";".join(ep_values)

    return {
        f"{field_prefix}_system_type": system_type,
        f"{field_prefix}_device": device,
        f"{field_prefix}_execution_provider": execution_provider,
        f"{field_prefix}_execution_providers": execution_providers,
    }


def _set_metadata_if_present(metadata: dict[str, Any], values: dict[str, Optional[str]]) -> None:
    for key, value in values.items():
        if value is not None:
            metadata.setdefault(key, value)


def _build_recipe_hash(run_config_json: dict[str, Any]) -> str:
    sanitized = deepcopy(run_config_json)
    _redact_recipe_hash_keys(sanitized)
    return hash_dict(sanitized)[:16]


def _redact_recipe_hash_keys(value: Any, key: Optional[str] = None) -> Any:
    if key in RECIPE_HASH_REDACTED_KEYS or _is_path_like_key(key):
        return RECIPE_HASH_REDACTED_VALUE
    if isinstance(value, dict):
        for child_key in list(value):
            value[child_key] = _redact_recipe_hash_keys(value[child_key], child_key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _redact_recipe_hash_keys(item, key)
    return value
