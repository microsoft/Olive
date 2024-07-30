# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from olive.common.hf.login import huggingface_login
from olive.logging import set_verbosity_from_env
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.passes.olive_pass import FullPassConfig

logger = logging.getLogger("olive")


def runner_entry(config, output_path, output_name):
    with open(config) as f:
        config_json = json.load(f)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_login(hf_token)

    model_json = config_json["model"]
    model = ModelConfig.from_json(model_json).create_model()

    pass_config = config_json["pass"]

    # Import the pass package configuration from the package_config
    package_config = OlivePackageConfig.load_default_config()
    package_config.import_pass_module(pass_config["type"])

    the_pass = FullPassConfig.from_json(pass_config).create_pass()
    output_model = the_pass.run(model, output_path)
    # save model json
    output_json = output_model.to_json()
    output_json_path = Path(output_path) / output_name
    with output_json_path.open("w") as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    set_verbosity_from_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="runner config")
    parser.add_argument("--output_path", help="Path of output model")
    parser.add_argument("--output_name", help="Name of output json file")

    args, _ = parser.parse_known_args()
    logger.info("command line arguments: %s", sys.argv)
    runner_entry(args.config, args.output_path, args.output_name)
