# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import sys

import olive.workflows.run as olive_run
from olive.common.utils import set_tempdir


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Distributed optimizations using Orchard")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pp", "tp"],
        help="Distribution mode",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        help="Number of GPUs in the group",
    )
    parser.add_argument(
        "--only_config",
        action="store_true",
        required=False,
        help="Whether to only dump the config file without running the optimization.",
    )
    parser.add_argument(
        "--tempdir",
        type=str,
        required=False,
        help="Root directory for tempfile directories and files",
    )

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # set tempdir
    set_tempdir(args.tempdir)

    json_file_template = "pp_template.json" if args.mode == "pp" else "tp_template.json"
    with open(json_file_template) as f:
        template_json = json.load(f)

    model_name = args.model_name
    # update model name
    template_json_str = json.dumps(template_json)
    template_json_str = template_json_str.replace("<model_name_placeholder>", model_name)
    template_json_str = template_json_str.replace("<nstages>", str(args.world_size))
    template_json_str = template_json_str.replace("<world_size>", str(args.world_size))
    template_json = json.loads(template_json_str)

    # dump config
    output_config_name = f"{args.mode}.json"
    with open(output_config_name, "w") as f:
        json.dump(template_json, f, indent=4)

    if not args.only_config:
        olive_run(template_json)  # pylint: disable=not-callable

    return 0


if __name__ == "__main__":
    sys.exit(main())
