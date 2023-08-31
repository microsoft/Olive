# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from olive.evaluator.metric import Metric
from olive.hardware import AcceleratorSpec
from olive.model import ModelConfig
from olive.systems.local import LocalSystem
from olive.systems.olive_system import OliveSystem
from olive.systems.utils import get_common_args


def parse_metric_args(raw_args):
    parser = argparse.ArgumentParser("Metric config")

    parser.add_argument("--metric_config", type=str, help="pass config", required=True)
    parser.add_argument("--metric_user_script", type=str, help="metric user script")
    parser.add_argument("--metric_script_dir", type=str, help="metric script dir")
    parser.add_argument("--metric_data_dir", type=str, help="metric data dir")

    return parser.parse_known_args(raw_args)


def parse_accelerator_args(raw_args):
    parser = argparse.ArgumentParser("Accelerator config")
    parser.add_argument("--accelerator_config", type=str, help="accelerator config", required=True)

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
    model_config, pipeline_output, extra_args = get_common_args(raw_args)
    metric_args, extra_args = parse_metric_args(extra_args)
    accelerator_args = parse_accelerator_args(extra_args)

    # load metric
    with open(metric_args.metric_config) as f:
        metric_config = json.load(f)
    metric = create_metric(metric_config, metric_args)

    # load model
    model = ModelConfig.from_json(model_config).create_model()

    with open(accelerator_args.accelerator_config) as f:
        accelerator_config = json.load(f)
    accelerator_spec = AcceleratorSpec(**accelerator_config)

    target: OliveSystem = LocalSystem()

    # metric result
    metric_result = target.evaluate_model(model, None, [metric], accelerator_spec)

    # save metric result json
    with open(Path(pipeline_output) / "metric_result.json", "w") as f:
        f.write(metric_result.json())


if __name__ == "__main__":
    main()
