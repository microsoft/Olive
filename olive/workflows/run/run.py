# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Union

from olive.workflows.run.config import RunConfig

logger = logging.getLogger(__name__)


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
    q_config["config"] = {"per_channel": "DEFAULT_SEARCH", "reduce_range": "DEFAULT_SEARCH"}
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


def run(config: Union[str, Path, dict]):
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
    # passes
    for pass_name, pass_config in config.passes.items():
        p = pass_config.create_pass()
        host = pass_config.host.create_system() if pass_config.host is not None else None
        evaluator = pass_config.evaluator.create_evaluator() if pass_config.evaluator is not None else None
        engine.register(p, pass_name, host, evaluator, pass_config.clean_run_cache)

    # run
    best_execution = engine.run(
        input_model, config.verbose, config.engine.output_dir, config.engine.output_name, config.engine.evaluation_only
    )
    logger.info(best_execution)
    return best_execution
