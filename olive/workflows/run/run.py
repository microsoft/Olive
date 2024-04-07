# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.metadata
import logging
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Union

from olive.auto_optimizer import AutoOptimizer
from olive.hardware.accelerator import create_accelerators
from olive.logging import enable_filelog, set_default_logger_severity, set_ort_logger_severity, set_verbosity_info
from olive.package_config import OlivePackageConfig
from olive.systems.common import SystemType
from olive.workflows.run.config import RunConfig, RunPassConfig

logger = logging.getLogger(__name__)


def dependency_setup(package_config: OlivePackageConfig, run_config: RunConfig):
    extras = deepcopy(package_config.extra_dependencies)

    def get_system_extras(host_type, accelerators, execution_providers):
        extra_name = None
        if host_type is None:
            extra_name = "cpu"
        elif host_type == SystemType.AzureML:
            extra_name = "azureml"
        elif host_type == SystemType.Docker:
            extra_name = "docker"
        elif host_type == SystemType.Local:
            if accelerators and "GPU" in list(map(str.upper, accelerators)):
                if execution_providers and "DmlExecutionProvider" in execution_providers:
                    extra_name = "directml"
                else:
                    extra_name = "gpu"
            else:
                extra_name = "cpu"

        return extra_name

    def get_pass_extras(pass_type):
        pass_module_config = package_config.passes.get(pass_type)

        extra_results = []
        extra_results.extend(pass_module_config.module_dependencies)
        for extra_name in pass_module_config.extra_dependencies:
            extra_results.extend(package_config.extra_dependencies.get(extra_name, []))
        return extra_results

    ort_packages = extras.get("ort", [])

    local_packages = []
    remote_packages = []

    # add dependencies for passes
    if run_config.passes:
        for pass_config in run_config.passes.values():
            host = pass_config.host or run_config.engine.host
            if (host and host.type == SystemType.Local) or not host:
                local_packages.extend(get_pass_extras(pass_config.type))
            else:
                remote_packages.extend(get_pass_extras(pass_config.type))
            if pass_config.type in ["SNPEConversion", "SNPEQuantization", "SNPEtoONNXConversion"]:
                logger.info(
                    "Please refer to https://microsoft.github.io/Olive/tutorials/passes/snpe.html to install SNPE"
                    " prerequisites for pass %s",
                    pass_config.type,
                )

    # add dependencies for engine
    host_type = None
    accelerators = []
    execution_providers = []
    if run_config.engine.host:
        host_type = run_config.engine.host.type
        if run_config.engine.host.config.accelerators:
            for acc in run_config.engine.host.config.accelerators:
                accelerators.append(acc.device)
                if acc.execution_providers:
                    execution_providers.extend(acc.execution_providers)

    system_extra_name = get_system_extras(host_type, accelerators, execution_providers)
    if system_extra_name:
        local_packages.extend(extras.get(system_extra_name))

    # install missing packages to local or tell user to install packages in their environment
    logger.info("The following packages are required in the local environment: %s", local_packages)
    packages_install = []
    for package in set(local_packages):
        if package in ort_packages:
            package_to_install = check_local_ort_installation(package)
            if package_to_install:
                packages_install.append(package_to_install)
        else:
            try:
                # use importlib.metadata to check if package is installed
                # better than __import__ since the package name can be different from the import name
                importlib.metadata.distribution(package)
                logger.info("%s is already installed.", package)
            except importlib.metadata.PackageNotFoundError:
                packages_install.append(package)

    if packages_install:
        # Install all packages once time
        cmd = [sys.executable, "-m", "pip", "install", *packages_install]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        logger.info("Successfully installed %s.", packages_install)

    if remote_packages:
        logger.info(
            "Please make sure the following packages are installed in %s environment: %s",
            run_config.engine.host.type,
            remote_packages,
        )


def run_engine(package_config: OlivePackageConfig, run_config: RunConfig, data_root: str = None):
    import onnxruntime as ort

    from olive.passes import Pass

    # for onnxruntime
    # ort_py_log_severity_level: python logging levels
    set_ort_logger_severity(run_config.engine.ort_py_log_severity_level)

    # ort_log_severity_level: C++ logging levels
    ort.set_default_logger_severity(run_config.engine.ort_log_severity_level)

    # input model
    input_model = run_config.input_model

    # Azure ML Client
    if run_config.azureml_client:
        run_config.engine.azureml_client_config = run_config.azureml_client

    engine = run_config.engine.create_engine()

    # run_config file will be uploaded to AML job
    is_azureml_system = (run_config.engine.host is not None and run_config.engine.host.type == SystemType.AzureML) or (
        run_config.engine.target is not None and run_config.engine.target.type == SystemType.AzureML
    )

    if is_azureml_system:
        from olive.systems.azureml.aml_system import AzureMLSystem

        AzureMLSystem.olive_config = run_config.to_json()

    no_evaluation = (
        engine.evaluator_config is None
        and run_config.passes
        and all(pass_config.evaluator is None for pass_config in run_config.passes.values())
    )
    accelerator_specs = create_accelerators(engine.target_config, skip_supported_eps_check=no_evaluation)

    pass_list = []
    acc_list = []
    if (
        not run_config.passes
        and run_config.auto_optimizer_config is not None
        and not run_config.auto_optimizer_config.disable_auto_optimizer
    ):
        # For auto optimizer, Olive generates passes and pass_flows for each accelerator
        # that means, the passes and pass_flows might be different for each accelerator
        for acc_spec in accelerator_specs:
            _passes, pass_flows = AutoOptimizer(
                input_model,
                engine.evaluator_config,
                acc_spec,
                run_config.auto_optimizer_config,
                run_config.data_configs,
            ).suggest()
            pass_list.append(({k: RunPassConfig.parse_obj(v) for k, v in _passes.items()}, pass_flows))
            acc_list.append([acc_spec])
    else:
        # For non-auto-optimizer case, Olive uses the same passes and pass_flows for all accelerators
        # if user needs different passes and pass_flows for each accelerator, they need to write multiple
        # config files.
        pass_list.append((run_config.passes, run_config.pass_flows))
        acc_list.append(accelerator_specs)

    run_rls = {}
    # Note that, in Olive, there are two positions where the accelerator_specs are looped over:
    # 1. olive workflow run level: this is where the accelerator_specs are created and passed to
    # the engine. In this level, accelerator specs can be used to generate passes and pass_flows.
    # 2. engine level: this is where the accelerator_specs are looped over to run the passes.
    # TODO(anyone): refactor the code to remove the engine level loop if possible.
    # For time being, we are keeping both loops, but in future, we might want to refactor the code
    # to remove engine level loop and pass the accelerator_specs to the engine directly.
    for accelerator_spec, (passes, pass_flows) in zip(acc_list, pass_list):
        engine.reset_passes()
        if passes:
            # First pass registers the necessary module implementation
            for pass_config in passes.values():
                logger.info("Importing pass module %s", pass_config.type)
                package_config.import_pass_module(pass_config.type)

            # Second pass, initializes the pass and registers it with the engine
            for pass_name, pass_config in passes.items():
                host = pass_config.host.create_system() if pass_config.host is not None else None
                engine.register(
                    Pass.registry[pass_config.type.lower()],
                    config=pass_config.config,
                    disable_search=pass_config.disable_search,
                    name=pass_name,
                    host=host,
                    evaluator_config=pass_config.evaluator,
                    clean_run_cache=pass_config.clean_run_cache,
                    output_name=pass_config.output_name,
                )
            engine.set_pass_flows(pass_flows)

        if data_root is None:
            data_root = run_config.data_root

        # run
        run_rls.update(
            engine.run(
                input_model,
                accelerator_spec,
                data_root,
                run_config.engine.packaging_config,
                run_config.engine.output_dir,
                run_config.engine.output_name,
                run_config.engine.evaluate_input_model,
            )
        )
    return run_rls


def run(
    run_config: Union[str, Path, dict],
    setup: bool = False,
    data_root: str = None,
    package_config: Union[str, Path, dict] = None,
):
    if package_config is None:
        package_config = OlivePackageConfig.get_default_config_path()

    # we use parse_file and parse_obj to be safe. If implemented as expected, both should be equivalent.
    if isinstance(package_config, (str, Path)):
        logger.info("Loading Olive module configuration from: %s", package_config)
        package_config = OlivePackageConfig.parse_file(package_config)
    else:
        package_config = OlivePackageConfig.parse_obj(package_config)

    if isinstance(run_config, (str, Path)):
        logger.info("Loading run configuration from: %s", run_config)
        run_config = RunConfig.parse_file(run_config)
    else:
        run_config = RunConfig.parse_obj(run_config)

    # set log level for olive
    set_default_logger_severity(run_config.engine.log_severity_level)
    if run_config.engine.log_to_file:
        enable_filelog(run_config.engine.log_severity_level)

    if setup:
        # set the log level to INFO for setup
        set_verbosity_info()
        dependency_setup(package_config, run_config)
        return None
    else:
        return run_engine(package_config, run_config, data_root)


def check_local_ort_installation(package_name: str):
    """Check whether ORT is installed. If not, will return current package name to install."""
    local_ort_packages = get_local_ort_packages()

    if not local_ort_packages:
        return package_name

    if "-" in package_name:
        night_package_name = f"ort-nightly-{package_name.split('-')[-1]}"
    else:
        night_package_name = "ort-nightly"

    if len(local_ort_packages) == 1 and local_ort_packages[0] in [package_name, night_package_name]:
        # only if one ort package is installed and it is the one we want
        # can be the stable or nightly version
        # TODO(jambayk): will probably be fine if we want cpu package but some other ort package is installed
        # but we can add a check for that if needed in the future
        logger.info("%s is already installed.", local_ort_packages[0])
        return None

    # instruction to user
    messages = [
        "There are one or more onnxruntime packages installed in your environment!",
        "The setup process is stopped to avoid potential conflicts. Please run the following commands manually:",
    ]
    uninstall_command = f"{sys.executable} -m pip uninstall -y " + " ".join(local_ort_packages)
    messages.append(f"Uninstall all existing onnxruntime packages: '{uninstall_command}'")
    messages.append(f"Install {package_name}: '{sys.executable} -m pip install {package_name}'")
    messages.append(
        "You can also instead install the corresponding nightly version following the instructions at"
        " https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages"
    )
    logger.warning("\n".join(messages))
    return None


def get_local_ort_packages() -> List[str]:
    all_packages = importlib.metadata.distributions()
    local_ort_packages = []
    for package in all_packages:
        package_name = package.metadata["Name"]
        if package_name == "onnxruntime-extensions":
            # onnxruntime-packages is under onnxruntime_extensions namespace
            # not an actual onnxruntime package
            continue
        if package_name.startswith(("onnxruntime", "ort-nightly")):
            local_ort_packages.append(package_name)
    return local_ort_packages
