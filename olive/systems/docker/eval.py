# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import logging
import os
import sys

from olive.common.hf.login import huggingface_login
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig

logger = logging.getLogger("olive")


def evaluate_entry(config, output_path, output_name, accelerator_type, execution_provider):
    with open(config) as f:
        config_json = json.load(f)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_login(hf_token)

    evaluator_config = OliveEvaluatorConfig(metrics=config_json["metrics"])
    model_json = config_json["model"]

    model = ModelConfig.from_json(model_json).create_model()

    evaluator: OliveEvaluator = evaluator_config.create_evaluator(model)
    metrics_res = evaluator.evaluate(
        model, evaluator_config.metrics, device=accelerator_type, execution_providers=execution_provider
    )

    with open(os.path.join(output_path, f"{output_name}"), "w") as f:
        f.write(metrics_res.json())
    logger.info("Metric result: %s", metrics_res)


if __name__ == "__main__":
    set_verbosity_from_env()

    parser = argparse.ArgumentParser()

    # no need to get model_path since it's already updated in config file
    parser.add_argument("--config", type=str, help="evaluation config")
    parser.add_argument("--output_path", help="Path of output model")
    parser.add_argument("--output_name", help="Name of output json file")
    parser.add_argument("--accelerator_type", type=str, help="accelerator type")
    parser.add_argument("--execution_provider", type=str, help="execution provider")

    args, _ = parser.parse_known_args()
    logger.info("command line arguments: %s", sys.argv)

    evaluate_entry(args.config, args.output_path, args.output_name, args.accelerator_type, args.execution_provider)
