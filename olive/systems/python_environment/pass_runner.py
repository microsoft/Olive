# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from olive.common.utils import set_tempdir
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.passes.olive_pass import FullPassConfig


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("--model_config", type=str, required=True, help="Path to input model json file")
    parser.add_argument("--pass_config", type=str, required=True, help="Path to pass json file")
    parser.add_argument("--tempdir", type=str, required=False, help="Root directory for tempfile directories and files")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to output model file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output model json file")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    set_verbosity_from_env()

    args = get_args(raw_args)

    set_tempdir(args.tempdir)

    model = ModelConfig.parse_file(args.model_config).create_model()
    pass_config = FullPassConfig.parse_file(args.pass_config)

    # Import the pass package configuration from the package_config
    package_config = OlivePackageConfig.load_default_config()
    package_config.import_pass_module(pass_config.type)

    the_pass = pass_config.create_pass()

    # run pass
    output_model = the_pass.run(model, args.output_model_path)

    # save model json
    with Path(args.output_path).open("w") as f:
        json.dump(output_model.to_json(), f, indent=4)


if __name__ == "__main__":
    main()
