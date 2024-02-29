# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json


def parse_common_args(raw_args):
    """Parse common args."""
    parser = argparse.ArgumentParser("Olive common args")

    # model args
    parser.add_argument("--model_config", type=str, help="Path to model config json", required=True)

    # pipeline output arg
    parser.add_argument("--pipeline_output", type=str, help="pipeline output path", required=True)

    return parser.parse_known_args(raw_args)


def parse_model_resources(model_resource_names, raw_args):
    """Parse model resources. These are the resources that were uploaded with the job."""
    parser = argparse.ArgumentParser("Model resources")

    # parse model resources
    for resource_name in model_resource_names:
        # will keep as required since only uploaded resources should be in this list
        parser.add_argument(f"--model_{resource_name}", type=str, required=True)

    return parser.parse_known_args(raw_args)


def get_common_args(raw_args):
    """Return the model_config.

    The return value includes json with the model resource paths filled in, the pipeline output path, and any
    extra args that were not parsed.
    """
    common_args, extra_args = parse_common_args(raw_args)

    # load model json
    with open(common_args.model_config) as f:
        model_json = json.load(f)
    # model json has a list of model resource names
    model_resource_names = model_json.pop("resource_names")
    model_resource_args, extra_args = parse_model_resources(model_resource_names, extra_args)

    for key, value in vars(model_resource_args).items():
        # remove the model_ prefix, the 1 is to only replace the first occurrence
        normalized_key = key.replace("model_", "", 1)
        model_json["config"][normalized_key] = value

    return model_json, common_args.pipeline_output, extra_args
