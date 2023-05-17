# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Union

import onnxruntime as ort

from olive import set_default_logger_severity
from olive.passes import Pass
from olive.systems.common import Device, SystemType
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
    with open(os.path.join(here, "../../extra_dependencies.json"), "r") as f:
        EXTRAS = json.load(f)
    DEPENDENCY_MAPPING = {
        "device": {
            SystemType.AzureML: EXTRAS.get("azureml"),
            SystemType.Docker: EXTRAS.get("docker"),
            SystemType.Local: {Device.CPU: EXTRAS.get("cpu"), Device.GPU: EXTRAS.get("gpu")},
        },
        "pass": {
            "OnnxFloatToFloat16": ["onnxconverter-common"],
            "OrtPerfTuning": ["psutil"],
            "QuantizationAwareTraining": ["pytorch-lightning"],
            "OpenVINOConversion": EXTRAS.get("openvino"),
            "OpenVINOQuantization": EXTRAS.get("openvino"),
            "IncQuantization": EXTRAS.get("inc"),
            "IncDynamicQuantization": EXTRAS.get("inc"),
            "IncStaticQuantization": EXTRAS.get("inc"),
        },
    }

    local_packages = []
    remote_packages = []

    # add dependencies for passes
    for _, pass_config in config.passes.items():
        host = pass_config.host or config.engine.host
        if (host and host.type == SystemType.Local) or not host:
            local_packages.extend(DEPENDENCY_MAPPING["pass"].get(pass_config.type, []))
        else:
            remote_packages.extend(DEPENDENCY_MAPPING["pass"].get(pass_config.type, []))
        if pass_config.type in ["SNPEConversion", "SNPEQuantization", "SNPEtoONNXConversion"]:
            logger.info(
                "Please refer to https://microsoft.github.io/Olive/tutorials/passes/snpe.html                 to"
                " install SNPE Prerequisites for pass {}".format(pass_config.type)
            )

    # add dependencies for engine
    if config.engine.host and config.engine.host.type == SystemType.Local:
        local_packages.extend(DEPENDENCY_MAPPING["device"][SystemType.Local][config.engine.host.config.device])
    elif not config.engine.host:
        local_packages.extend(DEPENDENCY_MAPPING["device"][SystemType.Local]["cpu"])
    else:
        local_packages.extend(DEPENDENCY_MAPPING["device"][config.engine.host.type])

    # install packages to local or tell user to install packages in their environment
    logger.info("The following packages will be installed: {}".format(" ".join(local_packages)))
    for package in set(local_packages):
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call(["python", "-m", "pip", "install", "{}".format(package)])
    if remote_packages:
        logger.info(
            "Please make sure the following packages are installed in {} environment: {}".format(
                config.engine.host.type, remote_packages
            )
        )


def run(config: Union[str, Path, dict], setup: bool = False):
    # we use parse_file and parse_obj to be safe. If implemented as expected, both should be equivalent.
    if isinstance(config, str) or isinstance(config, Path):
        config = RunConfig.parse_file(config)
    else:
        config = RunConfig.parse_obj(config)

    # set ort log level
    set_default_logger_severity(config.engine.log_severity_level)
    ort.set_default_logger_severity(config.engine.ort_log_severity_level)

    # input model
    input_model = config.input_model.create_model()

    # Azure ML Client
    if config.azureml_client:
        config.engine.azureml_client_config = config.azureml_client

    # engine
    engine = config.engine.create_engine()

    if (config.passes is None or not config.passes) and (not config.engine.evaluation_only):
        # TODO enhance this logic for more passes templates
        engine, config = automatically_insert_passes(config)

    if setup:
        dependency_setup(config)
    else:
        # passes
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
            )

        # run
        best_execution = engine.run(
            input_model,
            config.engine.packaging_config,
            config.verbose,
            config.engine.output_dir,
            config.engine.output_name,
            config.engine.evaluation_only,
        )
        return best_execution
