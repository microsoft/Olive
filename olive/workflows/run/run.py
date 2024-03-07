# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Union

from olive.auto_optimizer import AutoOptimizer
from olive.hardware.accelerator import create_accelerators
from olive.logging import enable_filelog, set_default_logger_severity, set_ort_logger_severity, set_verbosity_info
from olive.systems.common import SystemType
from olive.workflows.run.config import RunConfig, RunPassConfig

logger = logging.getLogger(__name__)


def dependency_setup(config):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "../../extra_dependencies.json")) as f:
        extras = json.load(f)

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
        pass_to_extra = {
            "OnnxFloatToFloat16": ["onnxconverter-common"],
            "OrtPerfTuning": ["psutil"],
            "QuantizationAwareTraining": ["pytorch-lightning"],
            "GptqQuantizer": ["auto-gptq", "optimum"],
        }

        pass_to_extra_names = {
            "OpenVINOConversion": ["openvino"],
            "OpenVINOQuantization": ["openvino"],
            "IncQuantization": ["inc"],
            "IncDynamicQuantization": ["inc"],
            "IncStaticQuantization": ["inc"],
            "OptimumConversion": ["optimum"],
            "OptimumMerging": ["optimum"],
            "TorchTRTConversion": ["torch-tensorrt"],
            "LoRA": ["lora"],
            "QLoRA": ["bnb", "lora"],
        }

        extra_results = []
        extra_results.extend(pass_to_extra.get(pass_type, []))
        for extra_name in pass_to_extra_names.get(pass_type, []):
            extra_results.extend(extras.get(extra_name))
        return extra_results

    ort_packages = ["onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-openvino"]

    local_packages = []
    remote_packages = []

    # add dependencies for passes
    if config.passes:
        for pass_config in config.passes.values():
            host = pass_config.host or config.engine.host
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
    if config.engine.target:
        host_type = config.engine.target.type
        if config.engine.target.config.accelerators:
            for acc in config.engine.target.config.accelerators:
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
            config.engine.target.type,
            remote_packages,
        )


def run_engine(config: RunConfig, data_root: str = None):
    import onnxruntime as ort

    from olive.passes import Pass

    # for onnxruntime
    # ort_py_log_severity_level: python logging levels
    set_ort_logger_severity(config.engine.ort_py_log_severity_level)

    # ort_log_severity_level: C++ logging levels
    ort.set_default_logger_severity(config.engine.ort_log_severity_level)

    # input model
    input_model = config.input_model

    # Azure ML Client
    if config.azureml_client:
        config.engine.azureml_client_config = config.azureml_client

    engine = config.engine.create_engine()

    # config file will be uploaded to AML job
    is_azureml_system = (config.engine.host is not None and config.engine.host.type == SystemType.AzureML) or (
        config.engine.target is not None and config.engine.target.type == SystemType.AzureML
    )

    if is_azureml_system:
        from olive.systems.azureml.aml_system import AzureMLSystem

        AzureMLSystem.olive_config = config.to_json()

    no_evaluation = engine.evaluator_config is None and all(
        pass_config.evaluator is None for pass_config in config.passes.values()
    )
    accelerator_specs = create_accelerators(engine.target_config, skip_supported_eps_check=no_evaluation)

    pass_list = []
    acc_list = []
    if (
        not config.passes
        and config.auto_optimizer_config is not None
        and not config.auto_optimizer_config.disable_auto_optimizer
    ):
        for acc_spec in accelerator_specs:
            _passes, pass_flows = AutoOptimizer(
                input_model,
                engine.evaluator_config,
                acc_spec,
                config.auto_optimizer_config,
                config.data_configs,
            ).suggest()
            pass_list.append(({k: RunPassConfig.parse_obj(v) for k, v in _passes.items()}, pass_flows))
            acc_list.append([acc_spec])
    else:
        pass_list.append((config.passes, config.pass_flows))
        acc_list.append(accelerator_specs)

    run_rls = {}
    for accelerator_spec, (passes, pass_flows) in zip(acc_list, pass_list):
        engine.reset_passes()
        if passes:
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
            data_root = config.data_root

        # run
        run_rls.update(
            engine.run(
                input_model,
                accelerator_spec,
                data_root,
                config.engine.packaging_config,
                config.engine.output_dir,
                config.engine.output_name,
                config.engine.evaluate_input_model,
            )
        )
    return run_rls


def run(config: Union[str, Path, dict], setup: bool = False, data_root: str = None):
    # we use parse_file and parse_obj to be safe. If implemented as expected, both should be equivalent.
    if isinstance(config, (str, Path)):
        config = RunConfig.parse_file(config)
    else:
        config = RunConfig.parse_obj(config)

    # set log level for olive
    set_default_logger_severity(config.engine.log_severity_level)
    if config.engine.log_to_file:
        enable_filelog(config.engine.log_severity_level)

    if setup:
        # set the log level to INFO for setup
        set_verbosity_info()
        dependency_setup(config)
        return None
    else:
        return run_engine(config, data_root)


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
