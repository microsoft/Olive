# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json

from olive.model import ModelConfig
from olive.passes.olive_pass import FullPassConfig


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("--model_json_path", type=str, required=True, help="Path to input model json file")
    parser.add_argument("--pass_json_path", type=str, required=True, help="Path to pass json file")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data root")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to output model file")
    parser.add_argument("--output_model_json_path", type=str, required=True, help="Path to output model json file")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    with open(args.model_json_path) as f:
        model_json = json.load(f)
    with open(args.pass_json_path) as f:
        pass_json = json.load(f)

    model = ModelConfig.from_json(model_json).create_model()
    the_pass = FullPassConfig.from_json(pass_json).create_pass()

    # run pass
    output_model = the_pass.run(model, args.data_root, args.output_model_path)
    # save model json
    output_json = output_model.to_json()

    with open(args.output_model_json_path, "w") as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    main()
