# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.metadata
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Union

import onnxruntime as ort

from olive.hardware import Device
from olive.logging import set_default_logger_severity, set_verbosity_info
from olive.passes import Pass
from olive.systems.common import SystemType
from olive.workflows.run.config import RunConfig

logger = logging.getLogger(__name__)


def automatically_insert_passes(config):
    new_config_dict = json.loads(config.json())
    new_passes = {}

    # insert onnx converter
    oc_config = {"type": "OnnxConversion"}
    new_passes["conversion"] = oc_config

    # insert transformer opt
    to_config = {"type": "OrtTransformersOptimization"}
    to_config["config"] = {"model_type": "bert"}
    to_config["disable_search"] = True
    new_passes["transformers_optimization"] = to_config

    # insert quantization
    q_config = {"type": "OnnxDynamicQuantization"}
    q_config["config"] = {"per_channel": "SEARCHABLE_VALUES", "reduce_range": "SEARCHABLE_VALUES"}
    q_config["clean_run_cache"] = False
    new_passes["dynamic_quantization"] = q_config

    # insert perf_tuning
    t_config = {"type": "OrtPerfTuning"}
    t_config["config"] = {"user_script": "user_script.py", "dataloader_func": "create_dataloader", "batch_size": 1}
    new_passes["perf_tuning"] = t_config

    new_config_dict["passes"] = new_passes
    new_config = RunConfig.parse_obj(new_config_dict)
    new_engine = new_config.engine.create_engine()

    return new_engine, new_config


def dependency_setup(config):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "../../extra_dependencies.json")) as f:  # noqa: PTH123
        extras = json.load(f)
    dependency_mapping = {
        "device": {
            SystemType.AzureML: extras.get("azureml"),
            SystemType.Docker: extras.get("docker"),
            SystemType.Local: {Device.CPU: extras.get("cpu"), Device.GPU: extras.get("gpu")},
        },
        "pass": {
            "OnnxFloatToFloat16": ["onnxconverter-common"],
            "OrtPerfTuning": ["psutil"],
            "QuantizationAwareTraining": ["pytorch-lightning"],
            "OpenVINOConversion": extras.get("openvino"),
            "OpenVINOQuantization": extras.get("openvino"),
            "IncQuantization": extras.get("inc"),
            "IncDynamicQuantization": extras.get("inc"),
            "IncStaticQuantization": extras.get("inc"),
            "OptimumConversion": extras.get("optimum"),
            "OptimumMerging": extras.get("optimum"),
            "TorchTRTConversion": extras.get("torch-tensorrt"),
        },
    }
    ort_packages = ["onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-openvino"]

    local_packages = []
    remote_packages = []

    # add dependencies for passes
    if config.passes:
        for pass_config in config.passes.values():
            host = pass_config.host or config.engine.host
            if (host and host.type == SystemType.Local) or not host:
                local_packages.extend(dependency_mapping["pass"].get(pass_config.type, []))
            else:
                remote_packages.extend(dependency_mapping["pass"].get(pass_config.type, []))
            if pass_config.type in ["SNPEConversion", "SNPEQuantization", "SNPEtoONNXConversion"]:
                logger.info(
                    "Please refer to https://microsoft.github.io/Olive/tutorials/passes/snpe.html to install SNPE"
                    f" prerequisites for pass {pass_config.type}"
                )

    # add dependencies for engine
    if config.engine.host and config.engine.host.type == SystemType.Local:
        # TODO(myguo): need to add DirectML support
        if config.engine.host.config.accelerators and "GPU" in list(
            map(str.upper, config.engine.host.config.accelerators)
        ):
            local_packages.extend(dependency_mapping["device"][SystemType.Local]["gpu"])
        else:
            local_packages.extend(dependency_mapping["device"][SystemType.Local]["cpu"])
    elif not config.engine.host:
        local_packages.extend(dependency_mapping["device"][SystemType.Local]["cpu"])
    else:
        local_packages.extend(dependency_mapping["device"][config.engine.host.type])

    # install missing packages to local or tell user to install packages in their environment
    logger.info(f"The following packages are required in the local environment: {local_packages}")
    for package in set(local_packages):
        if package in ort_packages:
            check_local_ort_installation(package)
        else:
            try:
                # use importlib.metadata to check if package is installed
                # better than __import__ since the package name can be different from the import name
                importlib.metadata.distribution(package)
                logger.info(f"{package} is already installed.")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"Installing {package}...")
                subprocess.check_call(["python", "-m", "pip", "install", f"{package}"])
                logger.info(f"Successfully installed {package}.")
    if remote_packages:
        logger.info(
            "Please make sure the following packages are installed in {} environment: {}".format(
                config.engine.host.type, remote_packages
            )
        )


def run(config: Union[str, Path, dict], setup: bool = False, data_root: str = None):
    # we use parse_file and parse_obj to be safe. If implemented as expected, both should be equivalent.
    if isinstance(config, (str, Path)):
        config = RunConfig.parse_file(config)
    else:
        config = RunConfig.parse_obj(config)

    # set ort log level
    set_default_logger_severity(config.engine.log_severity_level)
    ort.set_default_logger_severity(config.engine.ort_log_severity_level)

    # input model
    input_model = config.input_model

    # Azure ML Client
    if config.azureml_client:
        config.engine.azureml_client_config = config.azureml_client

    # engine
    engine = config.engine.create_engine()

    if not config.passes and not config.engine.evaluate_input_model:
        # TODO(trajep): enhance this logic for more passes templates
        engine, config = automatically_insert_passes(config)

    if setup:
        # set the log level to INFO for setup
        set_verbosity_info()
        dependency_setup(config)
        return None
    else:
        # passes
        if config.passes:
            for pass_name, pass_config in config.passes.items():
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
            engine.set_pass_flows(config.pass_flows)

        if data_root is None:
            data_root = config.data_root

        # run
        return engine.run(
            input_model,
            data_root,
            config.engine.packaging_config,
            config.engine.output_dir,
            config.engine.output_name,
            config.engine.evaluate_input_model,
        )


def check_local_ort_installation(package_name: str):
    local_ort_packages = get_local_ort_packages()

    if not local_ort_packages:
        # this case should not happen right now, for future proofing
        # onnxruntime is a dependency of olive so it must already be present
        # olive import would not have succeeded otherwise
        logger.info(f"Installing {package_name}...")
        subprocess.check_call(["python", "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}.")
        return

    if "-" in package_name:
        night_package_name = f"ort-nightly-{package_name.split('-')[-1]}"
    else:
        night_package_name = "ort-nightly"

    if len(local_ort_packages) == 1 and local_ort_packages[0] in [package_name, night_package_name]:
        # only if one ort package is installed and it is the one we want
        # can be the stable or nightly version
        # TODO(jambayk): will probably be fine if we want cpu package but some other ort package is installed
        # but we can add a check for that if needed in the future
        logger.info(f"{local_ort_packages[0]} is already installed.")
        return

    # instruction to user
    messages = [
        "There are one or more onnxruntime packages installed in your environment!",
        "Please run the following commands:",
    ]
    uninstall_command = "python -m pip uninstall -y " + " ".join(local_ort_packages)
    messages.append(f"Uninstall all existing onnxruntime packages: '{uninstall_command}'")
    messages.append(f"Install {package_name}: 'python -m pip install {package_name}'")
    messages.append(
        "You can also instead install the corresponding nightly version following the instructions at"
        " https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages"
    )
    logger.warning("\n".join(messages))


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
