# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from typing import Tuple

from olive.common.utils import set_nested_dict_value


def parse_common_args(raw_args):
    """Parse common args."""
    parser = argparse.ArgumentParser("Olive common args")

    # pipeline output arg
    parser.add_argument("--pipeline_output", type=str, help="pipeline output path", required=True)

    return parser.parse_known_args(raw_args)


def parse_config(raw_args, name: str) -> Tuple[dict, str]:
    """Parse config and related resource args."""
    parser = argparse.ArgumentParser(f"{name} config")

    # parse config arg
    config_name = f"{name}_config"
    resource_map_name = f"{name}_resource_map"
    parser.add_argument(f"--{config_name}", type=str, help=f"{name} config", required=True)
    parser.add_argument(f"--{resource_map_name}", type=str, help=f"{name} resource path")

    args, extra_args = parser.parse_known_args(raw_args)
    args = vars(args)

    # load config json
    with open(args[config_name]) as f:
        config = json.load(f)

    if args[resource_map_name] is None:
        return config, extra_args

    # load resource map json
    with open(args[resource_map_name]) as f:
        resource_map = json.load(f)

    # parse resources and replace in config
    parser = argparse.ArgumentParser(f"{name} resources")
    for resource_name in resource_map:
        parser.add_argument(f"--{resource_name}", type=str, help=f"{resource_name} path", required=True)

    args, extra_args = parser.parse_known_args(extra_args)
    args = vars(args)
    for resource_name, resource_key in resource_map.items():
        set_nested_dict_value(config, resource_key, args[resource_name])

    return config, extra_args


def get_common_args(raw_args):
    """Return the model_config.

    The return value includes json with the model resource paths filled in, the pipeline output path, and any
    extra args that were not parsed.
    """
    common_args, extra_args = parse_common_args(raw_args)

    model_json, extra_args = parse_config(extra_args, "model")

    return model_json, common_args.pipeline_output, extra_args
