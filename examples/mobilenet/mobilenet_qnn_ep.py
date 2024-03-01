# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import os
from pathlib import Path

from download_files import main as download_files
from onnxruntime import __version__ as OrtVersion
from packaging import version

import olive.workflows.run as olive_run


def get_eval_variables():
    env_vars = {"<qnn_env_path>": "QNN_ENV_PATH", "<qnn_lib_path>": "QNN_LIB_PATH"}
    for name in list(env_vars.keys()):
        try:
            env_vars[name] = Path(os.environ[env_vars[name]]).resolve().as_posix()
        except KeyError:
            raise ValueError(f"Environment variable {env_vars[name]} is not set") from None
    return env_vars


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="MobileNet Quantization for QNN EP")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--config_only", action="store_true", help="Generate config file only")
    parser.add_argument("--skip_data_download", action="store_true", help="Skip downloading data files")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    if version.parse(OrtVersion) < version.parse("1.17.0"):
        raise ValueError("This example requires ONNX Runtime 1.17.0 or later.")

    with open("mobilenet_qnn_ep_template.json") as f:
        config = json.load(f)

    config_name = "mobilenet_qnn_ep"
    if args.evaluate:
        template_args = get_eval_variables()
        template_args.update(
            {
                "<target>": "qnn_ep_env",
                "<evaluator>": "common_evaluator",
            }
        )
        # Set the environment variables
        config_str = json.dumps(config)
        for name, value in template_args.items():
            config_str = config_str.replace(name, value)
        config = json.loads(config_str)
        config_name += "_eval"
    else:
        # delete unnecessary fields
        del (
            config["systems"],
            config["evaluators"],
            config["engine"]["target"],
            config["engine"]["evaluator"],
            config["engine"]["evaluate_input_model"],
        )
        config_name += "_no_eval"

    if args.config_only:
        with open(f"{config_name}.json", "w") as f:
            json.dump(config, f, indent=4)
        print(f"Config file {config_name}.json is generated")  # noqa: T201
    else:
        if not args.skip_data_download:
            download_files()
        olive_run(config)  # pylint: disable=not-callable


if __name__ == "__main__":
    main()
