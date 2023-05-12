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
from olive.systems.local import LocalSystem
from olive.systems.utils import get_model_config, parse_common_args


def parse_metric_args(raw_args):
    parser = argparse.ArgumentParser("Metric config")

    parser.add_argument("--metric_config", type=str, help="pass config", required=True)
    parser.add_argument("--metric_user_script", type=str, help="metric user script")
    parser.add_argument("--metric_script_dir", type=str, help="metric script dir")
    parser.add_argument("--metric_data_dir", type=str, help="metric data dir")

    return parser.parse_args(raw_args)


def create_metric(metric_config, metric_args):
    for key, value in vars(metric_args).items():
        if key == "metric_config":
            continue
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
    # load metric
    metric_config = json.load(open(metric_args.metric_config))
    metric = create_metric(metric_config, metric_args)

    # HF model tokenizer will be loaded from HF model hub
    if model_config["config"].get("hf_config"):
        if not model_config["config"]["hf_config"].get("model_name"):
            print("model_name is not specified in hf_config. Skip updating model_name in pre_process_data component.")
        else:
            metric.data_config.components["pre_process_data"].params["model_name"] = model_config["config"][
                "hf_config"
            ]["model_name"]

    model = ModelConfig.from_json(model_config).create_model()

    # create_evaluator
    evaluator = OliveEvaluator([metric])

    target = LocalSystem()

    # metric result
    metric_result = evaluator.evaluate(model, target)

    # save metric result json
    json.dump(metric_result, open(Path(common_args.pipeline_output) / "metric_result.json", "w"))


if __name__ == "__main__":
    main()
