# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from olive.common.utils import set_tempdir
from olive.evaluator.metric import Metric
from olive.hardware import AcceleratorSpec
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig
from olive.systems.local import LocalSystem

if TYPE_CHECKING:
    from olive.systems.olive_system import OliveSystem


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("--model_config", type=str, required=True, help="Path to input model json file")
    parser.add_argument("--metrics_config", type=str, required=True, help="Path to metrics json file")
    parser.add_argument("--accelerator_config", type=str, required=True, help="Path to accelerator json file")
    parser.add_argument("--data_root", type=str, required=False, help="Path to data root")
    parser.add_argument("--tempdir", type=str, required=False, help="Root directory for tempfile directories and files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to metrics result json file")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    set_verbosity_from_env()

    args = get_args(raw_args)

    set_tempdir(args.tempdir)

    model_config = ModelConfig.parse_file(args.model_config)
    metrics = []
    with Path(args.metrics_config).open() as f:
        metrics_json = json.load(f)
        metrics = [Metric.parse_obj(metric_json) for metric_json in metrics_json]
    with Path(args.accelerator_config).open() as f:
        accelerator_config = json.load(f)
    accelerator_spec = AcceleratorSpec(**accelerator_config)

    target: OliveSystem = LocalSystem()

    # metric result
    metric_result = target.evaluate_model(model_config, args.data_root, metrics, accelerator_spec)

    # save metric result json
    with Path(args.output_path).open("w") as f:
        f.write(metric_result.json())


if __name__ == "__main__":
    main()
