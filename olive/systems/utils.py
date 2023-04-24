# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from olive.model import ModelType


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Olive model args")

    # model args
    parser.add_argument("--model_config", type=str, help="model config", required=True)
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_script", type=str, help="model script")
    parser.add_argument("--model_script_dir", type=str, help="model script dir")

    # pipeline output arg
    # model output args
    parser.add_argument("--pipeline_output", type=str, help="pipeline output path", required=True)

    return parser.parse_known_args(raw_args)


def get_model_config(common_args):
    model_json = json.load(open(common_args.model_config))

    for key, value in common_args.__dict__.items():
        if value and key in model_json["config"]:
            model_json["config"][key] = value
    if model_json["type"].lower() == "onnxmodel" and model_json["config"]["model_type"] == ModelType.LocalFolder:
        model_json["config"]["model_path"] = str(Path(model_json["config"]["model_path"]) / "model.onnx")

    return model_json
