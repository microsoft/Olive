# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from olive.cache import isolated_cache_env
from olive.common.utils import set_tempdir
from olive.hardware.constants import ExecutionProvider
from olive.logging import set_default_logger_severity, set_ort_logger_severity, set_verbosity_info
from olive.package_config import OlivePackageConfig
from olive.systems.accelerator_creator import create_accelerator
from olive.systems.common import SystemType
from olive.workflows.run.builds import get_build_cache_dir, parse_run_config
from olive.workflows.run.config import RunConfig

if TYPE_CHECKING:
    from olive.engine.config import RunPassConfig

logger = logging.getLogger(__name__)


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
        import onnxruntime as ort

        ort.set_default_logger_severity(run_config.engine.ort_log_severity_level)
    except Exception:
        logger.warning("ORT log severity level configuration ignored since the module isn't installed.")

    olive_config = run_config.to_json()
    engine = run_config.engine.create_engine(package_config, workflow_id)
    engine.cache.cache_olive_config(olive_config)

    # check if target is not used
    used_passes_configs = get_used_passes_configs(run_config)
    target_not_used = engine.evaluator_config is None and all(
        pass_config.evaluator is None and not get_run_on_target(package_config, pass_config)
        for pass_config in used_passes_configs
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
):
    # set tempdir
    set_tempdir(tempdir)

    if package_config is None:
        package_config = OlivePackageConfig.get_default_config_path()

    package_config = OlivePackageConfig.parse_file_or_obj(package_config)
    parsed_config = parse_run_config(run_config)
    if isinstance(parsed_config, dict):
        if list_required_packages:
            _list_required_packages(package_config, parsed_config.values())
            return None
        return _run_builds_in_parallel(package_config, parsed_config)

    if list_required_packages:
        _list_required_packages(package_config, [parsed_config])
        return None
    return _run_single(package_config, parsed_config)


def _run_builds_in_parallel(package_config: OlivePackageConfig, build_configs: dict[str, RunConfig]) -> OrderedDict:
    results = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=len(build_configs), thread_name_prefix="olive-build") as executor:
        future_to_name = {
            executor.submit(_run_named_build, deepcopy(package_config), build_name, build_config): build_name
            for build_name, build_config in build_configs.items()
        }
        for future in as_completed(future_to_name):
            build_name = future_to_name[future]
            try:
                results[build_name] = future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.setdefault(build_name, []).append(exc)

    if errors:
        failed_names = [build_name for build_name in build_configs if build_name in errors]
        first_failed = failed_names[0]
        details = "; ".join(
            f"{build_name}: {type(error).__name__}: {error}"
            for build_name in failed_names
            for error in errors[build_name]
        )
        raise RuntimeError(f"Build(s) {failed_names} failed: {details}") from errors[first_failed][0]

    return OrderedDict((build_name, results[build_name]) for build_name in build_configs)


def _run_named_build(package_config: OlivePackageConfig, build_name: str, run_config: RunConfig):
    logger.info("Running build %s", build_name)
    with isolated_cache_env(get_build_cache_dir(run_config)):
        return _run_single(package_config, run_config)


def _run_single(package_config: OlivePackageConfig, run_config: RunConfig):
    if run_config.engine.host and run_config.engine.host.type == SystemType.Docker:
        docker_system = run_config.engine.host.create_system()
        return docker_system.run_workflow(run_config)

    set_default_logger_severity(run_config.engine.log_severity_level)
    return run_engine(package_config, run_config)


def _list_required_packages(package_config: OlivePackageConfig, run_configs) -> None:
    set_verbosity_info()
    required_packages = set()
    for run_config in run_configs:
        required_packages.update(get_required_packages(package_config, run_config))
    generate_files_from_packages(required_packages, "olive_requirements.txt")


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
