# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.model import ModelConfig
from olive.systems.utils import get_model_config, parse_common_args


def parse_metric_args(raw_args):
    parser = argparse.ArgumentParser("Metric config")

    parser.add_argument("--metric_config", type=str, help="pass config", required=True)
    parser.add_argument("--metric_user_script", type=str, help="metric user script", required=True)
    parser.add_argument("--metric_script_dir", type=str, help="metric script dir")
    parser.add_argument("--metric_data_dir", type=str, help="metric data dir")

    return parser.parse_args(raw_args)


def create_metric(metric_config, metric_args):
    for key, value in vars(metric_args).items():
        if value is not None:
            key = key.replace("metric_", "")
            metric_config["user_config"][key] = value

    p = Metric.from_json(metric_config)
    return p


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)
    metric_args = parse_metric_args(extra_args)

    # load model
    model_config = get_model_config(common_args)
    model = ModelConfig.from_json(model_config).create_model()

    # load metric
    metric_config = json.load(open(metric_args.metric_config))
    metric = create_metric(metric_config, metric_args)

    # create_evaluator
    evaluator = OliveEvaluator([metric])

    # metric result
    metric_result = evaluator.evaluate(model)

    # save metric result json
    json.dump(metric_result, open(Path(common_args.pipeline_output) / "metric_result.json", "w"))


if __name__ == "__main__":
    main()
