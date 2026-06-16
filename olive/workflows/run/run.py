# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from olive.common.utils import set_tempdir
from olive.hardware.constants import ExecutionProvider
from olive.logging import set_default_logger_severity, set_ort_logger_severity, set_verbosity_info
from olive.package_config import OlivePackageConfig
from olive.systems.accelerator_creator import create_accelerator
from olive.systems.common import SystemType
from olive.workflows.run.config import RunConfig, RunEngineConfig

if TYPE_CHECKING:
    from olive.engine.config import BuildConfig, RunPassConfig

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

    if run_config.builds:
        return _run_builds(package_config, run_config, olive_config)

    engine = run_config.engine.create_engine(package_config, workflow_id)
    engine.cache.cache_olive_config(olive_config)

    # check if target is not used
    used_passes_configs = get_used_passes_configs(run_config)
    target_not_used = _compute_target_not_used(package_config, engine.evaluator_config, used_passes_configs)

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


def _run_builds(package_config: OlivePackageConfig, run_config: RunConfig, olive_config: dict) -> dict:
    """Run every entry in ``run_config.builds`` as an independent workflow.

    Returns a ``dict[str, WorkflowOutput]`` keyed by build name. Each build gets its own engine,
    pipeline subset (from ``passes`` in the order declared by ``pipeline``), input model slice
    (via ``select_components`` when ``components`` is set) and host/target/evaluator overrides.
    """
    _validate_build_components(run_config)
    workflow_id = run_config.workflow_id
    outputs: dict = OrderedDict()
    for build_name, build in run_config.builds.items():
        logger.info("Running build %s", build_name)
        engine_config = _make_build_engine_config(run_config, build)
        engine = engine_config.create_engine(package_config, f"{workflow_id}_{build_name}")
        engine.cache.cache_olive_config(olive_config)

        pipeline_subset: dict[str, list[RunPassConfig]] = OrderedDict()
        for pass_name in build.pipeline:
            # deepcopy so each build engine owns its pass configs; Engine.initialize mutates them in place.
            pipeline_subset[pass_name] = deepcopy(run_config.passes[pass_name])

        input_model = run_config.input_model
        if build.components:
            input_model = input_model.select_components(build.components)

        used_passes_configs = [p for passes in pipeline_subset.values() for p in passes]
        target_not_used = _compute_target_not_used(package_config, engine.evaluator_config, used_passes_configs)
        is_ep_required = _is_execution_provider_required_for_passes(package_config, used_passes_configs) or (
            engine.evaluator_config is not None
            and engine_config.evaluate_input_model
            and input_model.type.lower() == "onnxmodel"
        )
        accelerator_spec = create_accelerator(
            engine.target_config, skip_supported_eps_check=target_not_used, is_ep_required=is_ep_required
        )

        engine.set_input_passes_configs(pipeline_subset)
        outputs[build_name] = engine.run(
            input_model,
            accelerator_spec,
            engine_config.packaging_config,
            build.output_dir,
            engine_config.evaluate_input_model,
            engine_config.log_to_file,
            engine_config.log_severity_level,
        )
    return outputs


def _validate_build_components(run_config: RunConfig) -> None:
    """Verify ``build.components`` names exist in the input model's selectable components."""
    needs_component_check = any(build.components for build in run_config.builds.values())
    if not needs_component_check:
        return
    available = run_config.input_model.get_components()
    if available is None:
        bad = [name for name, build in run_config.builds.items() if build.components]
        raise ValueError(
            f"Builds {bad} declare `components` but the input model is not a CompositeModel"
            f" (got type {run_config.input_model.type!r})."
        )
    for build_name, build in run_config.builds.items():
        if not build.components:
            continue
        missing = [n for n in build.components if n not in available]
        if missing:
            raise ValueError(
                f"Build {build_name!r} references unknown component(s) {missing}. Available components: {available}."
            )


def _make_build_engine_config(run_config: RunConfig, build: "BuildConfig") -> RunEngineConfig:
    """Clone the parent engine config and apply this build's host/target/evaluator/search overrides."""
    engine_dump = run_config.engine.model_dump()
    systems = run_config.systems or {}
    evaluators = run_config.evaluators or {}
    if build.host is not None:
        engine_dump["host"] = _resolve_build_ref(build.host, systems, "system")
    if build.target is not None:
        engine_dump["target"] = _resolve_build_ref(build.target, systems, "system")
    if build.evaluator is not None:
        engine_dump["evaluator"] = _resolve_build_ref(build.evaluator, evaluators, "evaluator")
    if build.search_strategy is not None:
        # search_strategy can be a bool or a SearchStrategyConfig; pydantic round-trips either form.
        ss = build.search_strategy
        engine_dump["search_strategy"] = ss if isinstance(ss, bool) else ss.model_dump()
    return RunEngineConfig.model_validate(engine_dump)


def _resolve_build_ref(value, registry: dict, label: str):
    """Resolve a string reference into the named entry; pass through dict/model instances unchanged."""
    if isinstance(value, str):
        if value not in registry:
            raise ValueError(f"Unknown {label} reference {value!r}. Known: {sorted(registry)}.")
        entry = registry[value]
        return entry.model_dump() if hasattr(entry, "model_dump") else deepcopy(entry)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return deepcopy(value)


def _compute_target_not_used(
    package_config: OlivePackageConfig,
    evaluator_config,
    pass_configs: list["RunPassConfig"],
) -> bool:
    return evaluator_config is None and all(
        pc.evaluator is None and not get_run_on_target(package_config, pc) for pc in pass_configs
    )


def _is_execution_provider_required_for_passes(
    package_config: OlivePackageConfig, pass_configs: list["RunPassConfig"]
) -> bool:
    return any(package_config.is_onnx_module(pc.type) for pc in pass_configs)


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
    run_config: RunConfig = RunConfig.parse_file_or_obj(run_config)

    if list_required_packages:
        # set the log level to INFO for packages
        set_verbosity_info()
        required_packages = get_required_packages(package_config, run_config)
        generate_files_from_packages(required_packages, "olive_requirements.txt")
        return None

    if run_config.engine.host and run_config.engine.host.type == SystemType.Docker:
        docker_system = run_config.engine.host.create_system()
        return docker_system.run_workflow(run_config)

    # set log level for olive
    set_default_logger_severity(run_config.engine.log_severity_level)
    return run_engine(package_config, run_config)


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
