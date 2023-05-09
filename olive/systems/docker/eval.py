# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import logging
import os
import sys

from olive.evaluator.evaluation import evaluator_adaptor
from olive.evaluator.metric import MetricList
from olive.evaluator.metric_config import SignalResult
from olive.model import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(module)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def evaluate_entry(config, model_path, output_path, output_name):
    with open(config, "r") as f:
        config_json = json.load(f)
    metric_list = MetricList(__root__=json.loads(config_json["metrics"])).__root__
    logger.info(f"Evaluation metric list: {metric_list}")

    model_json = config_json["model"]
    model_json["config"]["model_path"] = model_path
    model = ModelConfig.from_json(model_json).create_model()

    metrics_res = {}
    for metric in metric_list:
        evaluator = evaluator_adaptor(metric)
        metrics_res[metric.name] = evaluator(model, metric)
    signal = SignalResult(signal=metrics_res)

    with open(os.path.join(output_path, f"{output_name}"), "w") as f:
        json.dump(signal.dict(), f)
    logger.info(f"Metric result: {signal.json()}")


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
