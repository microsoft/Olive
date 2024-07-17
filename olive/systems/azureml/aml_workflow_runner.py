# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path

from olive.common.constants import WORKFLOW_ARTIFACTS, WORKFLOW_CONFIG
from olive.common.utils import aml_runner_hf_login, copy_dir
from olive.logging import set_verbosity_from_env
from olive.systems.utils.arg_parser import parse_config
from olive.workflows import run as olive_run


def parse_workflow_config(raw_args):
    parser = argparse.ArgumentParser("Olive workflow config")

    parser.add_argument(f"--{WORKFLOW_CONFIG}", type=str, help="olive workflow config", required=True)
    parser.add_argument(f"--{WORKFLOW_ARTIFACTS}", type=str, help="olive workflow artifacts", required=True)

    return parser.parse_known_args(raw_args)


def parse_data_config(olive_config, extra_args):
    if "data_configs" not in olive_config or not olive_config["data_configs"]:
        return olive_config, extra_args

    new_data_config = []
    for data_config in olive_config["data_configs"]:
        parsed_data_config, extra_args = parse_config(extra_args, data_config.get("name"))
        new_data_config.append(parsed_data_config)
    olive_config["data_configs"] = new_data_config
    return olive_config, extra_args


def parse_dict_resources_config(resources, olive_config, extra_args):
    if resources not in olive_config or not olive_config[resources]:
        return olive_config, extra_args

    new_resources_config = {}
    for resource_name in olive_config[resources]:
        resource_config, extra_args = parse_config(extra_args, resource_name)
        new_resources_config[resource_name] = resource_config
    olive_config[resources] = new_resources_config
    return olive_config, extra_args


def main(raw_args=None):
    set_verbosity_from_env()

    # login to hf if HF_LOGIN is set to True
    aml_runner_hf_login()

    olive_config_args, extra_args = parse_workflow_config(raw_args)
    model_json, extra_args = parse_config(extra_args, "model")

    with open(olive_config_args.workflow_config) as f:
        olive_config = json.load(f)

    olive_config["input_model"] = model_json
    olive_config, extra_args = parse_data_config(olive_config, extra_args)

    dict_resources_list = ["passes", "evaluators", "systems"]
    for resources in dict_resources_list:
        olive_config, extra_args = parse_dict_resources_config(resources, olive_config, extra_args)

    olive_run(olive_config, setup=True)
    olive_run(olive_config)

    workflow_artifacts_path = Path(olive_config_args.workflow_artifacts)

    remote_cache_dir = workflow_artifacts_path / "cache"
    remote_output_dir = workflow_artifacts_path / "output"

    # Clean up the remote cache and output directories if they exist
    if remote_cache_dir.exists():
        shutil.rmtree(remote_cache_dir)
    if remote_output_dir.exists():
        shutil.rmtree(remote_output_dir)

    copy_dir(olive_config["engine"]["cache_dir"], workflow_artifacts_path / "cache")
    copy_dir(olive_config["engine"]["output_dir"], workflow_artifacts_path / "output")


if __name__ == "__main__":
    main()
