# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import logging
import os
import sys

from olive.evaluator.evaluation import evaluate_accuracy, evaluate_custom_metric, evaluate_latency
from olive.evaluator.metric import MetricList, MetricType
from olive.model import ModelConfig

logger = logging.getLogger(__name__)


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
        if metric.name == MetricType.ACCURACY:
            metrics_res[MetricType.ACCURACY] = evaluate_accuracy(model, metric)
        elif metric.name == MetricType.LATENCY:
            metrics_res[MetricType.LATENCY] = evaluate_latency(model, metric)
        else:
            metrics_res[metric.name] = evaluate_custom_metric(model, metric)

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
