# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from typing import Tuple

from olive.common.utils import set_nested_dict_value


def parse_resources_args(raw_args):
    """Parse resources args.

    Get the resources.
    Resources are expected to be provided as inputs of the form --resource__{i} where i is the index of the resource.
    `num_resources` is the number of resources provided.
    """
    parser = argparse.ArgumentParser("Olive common args")

    # resources arg
    parser.add_argument("--num_resources", type=int, help="number of resources", required=True)

    args, extra_args = parser.parse_known_args(raw_args)

    if args.num_resources == 0:
        return {}, extra_args

    # parse resources
    parser = argparse.ArgumentParser("Olive resources")
    for i in range(args.num_resources):
        parser.add_argument(f"--resource__{i}", type=str, help=f"resource {i} path", required=True)

    resource_args, extra_args = parser.parse_known_args(extra_args)

    return vars(resource_args), extra_args


def parse_config(raw_args, name: str, resources: dict) -> Tuple[dict, str]:
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

    # replace resource paths in config
    for resource_key, resource_name in resource_map:
        set_nested_dict_value(config, resource_key, resources[resource_name])

    return config, extra_args


def parse_pipeline_output(raw_args):
    """Parse the pipeline output arg."""
    parser = argparse.ArgumentParser("Pipeline output")

    parser.add_argument("--pipeline_output", type=str, help="pipeline output path", required=True)

    args, extra_args = parser.parse_known_args(raw_args)

    return args.pipeline_output, extra_args


def get_common_args(raw_args):
    """Return the model_config.

    The return value includes json with the model resource paths filled in, the pipeline output path, and any
    extra args that were not parsed.
    """
    resources, extra_args = parse_resources_args(raw_args)

    pipeline_output, extra_args = parse_pipeline_output(extra_args)
    model_json, extra_args = parse_config(extra_args, "model", resources)

    return pipeline_output, resources, model_json, extra_args
