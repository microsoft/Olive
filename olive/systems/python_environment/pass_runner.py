# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import sys
from pathlib import Path

from olive.model import ModelConfig
from olive.passes.olive_pass import FullPassConfig

ort_inference_utils_parent = Path(__file__).resolve().parent.parent.parent / "common"
sys.path.append(str(ort_inference_utils_parent))

# ruff: noqa: PTH123


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("--model_json_path", type=str, required=True, help="Path to input model json file")
    parser.add_argument("--pass_json_path", type=str, required=True, help="Path to pass json file")
    parser.add_argument("--point_json_path", type=str, default=None, help="Path to point json file")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data root")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to output model file")
    parser.add_argument("--output_model_json_path", type=str, required=True, help="Path to output model json file")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)
    args.model_json_path = Path(args.model_json_path)
    args.pass_json_path = Path(args.pass_json_path)
    args.output_model_path = Path(args.output_model_path)
    args.output_model_json_path = Path(args.output_model_json_path)

    with open(args.model_json_path) as f:
        model_json = json.load(f)
    with open(args.pass_json_path) as f:
        pass_json = json.load(f)

    if args.point_json_path:
        args.point_json_path = Path(args.point_json_path)
        with open(args.point_json_path) as f:
            point = json.load(f)
    else:
        point = None

    if args.data_root:
        args.data_root = Path(args.data_root)
    else:
        args.data_root = None

    model = ModelConfig.from_json(model_json).create_model()
    the_pass = FullPassConfig.from_json(pass_json).create_pass()

    # run pass
    output_model = the_pass.run(model, args.data_root, args.output_model_path, point)

    # save model json
    output_json = output_model.to_json()

    with args.output_model_json_path.open("w") as f:
        json.dump(output_json, f, indent=4)

    return args.output_model_json_path


if __name__ == "__main__":
    main()
