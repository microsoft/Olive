# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import subprocess
from pathlib import Path
from typing import Union

from olive.systems.common import Device, SystemType
from olive.workflows.run.config import RunConfig

logger = logging.getLogger(__name__)


DEPENDENCY_MAPPING = {
    "device": {
        SystemType.AzureML: ["azure-ai-ml>=0.1.0b6", "azure-identity"],
        SystemType.Docker: ["docker"],
        SystemType.Local: {Device.CPU: ["onnxruntime==1.13.1"], Device.GPU: ["onnxruntime-gpu==1.13.1"]},
    },
    "pass": {
        "OnnxFloatToFloat16": ["onnxconverter-common"],
        "OrtPerfTuning": ["psutil"],
        "QuantizationAwareTraining": ["pytorch-lightning"],
        "OpenVINOConversion": ["openvino==2022.3.0", "openvino-dev[tensorflow,onnx]==2022.3.0"],
        "OpenVINOQuantization": ["openvino==2022.3.0", "openvino-dev[tensorflow,onnx]==2022.3.0"],
    },
}


def automatically_insert_passes(config):
    new_config_dict = json.loads(config.json())
    new_passes = {}

    # insert onnx converter
    oc_config = {"type": "OnnxConversion"}
    oc_config["config"] = config.engine.model_io_config
    new_passes["conversion"] = oc_config

    # insert transformer opt
    to_config = {"type": "OnnxTransformersOptimization"}
    to_config["config"] = {"model_type": "bert"}
    new_passes["transformers_optimization"] = to_config

    # insert quantization
    q_config = {"type": "OnnxDynamicQuantization"}
    q_config["config"] = {"per_channel": "SEARCHABLE_VALUES", "reduce_range": "SEARCHABLE_VALUES"}
    q_config["clean_run_cache"] = False
    new_passes["dynamic_quantization"] = q_config

    # insert thread_tuning
    t_config = {"type": "OnnxThreadTuning"}
    t_config["config"] = {"user_script": "user_script.py", "dataloader_func": "create_dataloader", "batch_size": 1}
    new_passes["thread_tuning"] = t_config

    new_config_dict["passes"] = new_passes
    new_config = RunConfig.parse_obj(new_config_dict)
    new_engine = new_config.engine.create_engine()

    return new_engine, new_config


def dependency_setup(config):
    required_packages = []

    for _, pass_config in config.passes.items():
        if DEPENDENCY_MAPPING["pass"].get(pass_config.type):
            required_packages.extend(DEPENDENCY_MAPPING["pass"].get(pass_config.type))
        if pass_config.type in ["SNPEConversion", "SNPEQuantization", "SNPEtoONNXConversion"]:
            logger.info(
                "Please refer to https://microsoft.github.io/Olive/tutorials/passes/snpe.html \
                to install SNPE Prerequisites for pass {}".format(
                    pass_config.type
                )
            )

    if config.engine.host.type == SystemType.Local:
        required_packages.extend(
            DEPENDENCY_MAPPING["device"][config.engine.host.type][config.engine.host.config.device]
        )
        logger.info("The following packages will be installed: {}".format(" ".join(required_packages)))
        for package in set(required_packages):
            try:
                __import__(package)
            except (ImportError):
                subprocess.check_call(["pip", "install", "{}".format(package)])
    else:
        for package in set(DEPENDENCY_MAPPING["device"][config.engine.host.type]):
            try:
                __import__(package)
            except (ImportError):
                subprocess.check_call(["pip", "install", "{}".format(package)])
        logger.info(
            "Please make sure the following packages are installed in {} environment: {}".format(
                config.engine.host.type, required_packages
            )
        )


def run(config: Union[str, Path, dict], setup: bool = False):
    # we use parse_file and parse_obj to be safe. If implemented as expected, both should be equivalent.
    if isinstance(config, str) or isinstance(config, Path):
        config = RunConfig.parse_file(config)
    else:
        config = RunConfig.parse_obj(config)

    # input model
    input_model = config.input_model.create_model()

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
            p = pass_config.create_pass()
            host = pass_config.host.create_system() if pass_config.host is not None else None
            evaluator = pass_config.evaluator.create_evaluator() if pass_config.evaluator is not None else None
            engine.register(p, pass_name, host, evaluator, pass_config.clean_run_cache)

        # run
        best_execution = engine.run(
            input_model,
            config.verbose,
            config.engine.output_dir,
            config.engine.output_name,
            config.engine.evaluation_only,
        )
        return best_execution
