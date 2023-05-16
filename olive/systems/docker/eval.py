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


def evaluate_entry(config, model_path, output_path, output_name):
    with open(config, "r") as f:
        config_json = json.load(f)
    evaluator_config = OliveEvaluatorConfig(metrics=config_json["metrics"])
    model_json = config_json["model"]

    if model_path != "None":
        model_json["config"]["model_path"] = model_path
    model = ModelConfig.from_json(model_json).create_model()

    evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
    metrics_res = evaluator.evaluate(model, evaluator_config.metrics)

    with open(os.path.join(output_path, f"{output_name}"), "w") as f:
        json.dump(metrics_res, f)
    logger.info(f"Metric result: {metrics_res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="evaluation config")
    parser.add_argument("--model_path", help="The input directory.")
    parser.add_argument("--output_path", help="Path of output model")
    parser.add_argument("--output_name", help="Name of output json file")

    args, _ = parser.parse_known_args()
    logger = logging.getLogger("module")
    logger.info("command line arguments: ", sys.argv)

    evaluate_entry(args.config, args.model_path, args.output_path, args.output_name)
