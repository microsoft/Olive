# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import logging
import os
import sys

from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig, OliveEvaluatorFactory
from olive.model import ModelConfig

logger = logging.getLogger(__name__)


def evaluate_entry(config, output_path, output_name, accelerator_type, execution_provider):
    with open(config, "r") as f:
        config_json = json.load(f)
    evaluator_config = OliveEvaluatorConfig(metrics=config_json["metrics"])
    model_json = config_json["model"]

    model = ModelConfig.from_json(model_json).create_model()

    evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
    metrics_res = evaluator.evaluate(
        model, None, evaluator_config.metrics, device=accelerator_type, execution_providers=execution_provider
    )

    with open(os.path.join(output_path, f"{output_name}"), "w") as f:
        f.write(metrics_res.json())
    logger.info(f"Metric result: {metrics_res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # no need to get model_path since it's already updated in config file
    parser.add_argument("--config", type=str, help="evaluation config")
    parser.add_argument("--output_path", help="Path of output model")
    parser.add_argument("--output_name", help="Name of output json file")
    parser.add_argument("--accelerator_type", type=str, help="accelerator type")
    parser.add_argument("--execution_provider", type=str, help="execution provider")

    args, _ = parser.parse_known_args()
    logger = logging.getLogger("module")
    logger.info("command line arguments: ", sys.argv)

    evaluate_entry(args.config, args.output_path, args.output_name, args.accelerator_type, args.execution_provider)
