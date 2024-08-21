# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from datetime import datetime
from pathlib import Path

from olive.common.constants import WORKFLOW_ARTIFACTS, WORKFLOW_CONFIG
from olive.common.hf.login import aml_runner_hf_login
from olive.common.utils import copy_dir
from olive.logging import set_verbosity_from_env
from olive.systems.utils.arg_parser import parse_config, parse_resources_args
from olive.workflows import run as olive_run

# ruff: noqa: T201


def parse_artifacts_config(raw_args):
    parser = argparse.ArgumentParser("Olive workflow artifacts config")
    parser.add_argument(f"--{WORKFLOW_ARTIFACTS}", type=str, help="olive workflow artifacts", required=True)

    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    set_verbosity_from_env()

    # login to hf if HF_LOGIN is set to True
    aml_runner_hf_login()

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    workflow_artifacts_args, extra_args = parse_artifacts_config(raw_args)
    resources, extra_args = parse_resources_args(raw_args)
    olive_config, extra_args = parse_config(extra_args, WORKFLOW_CONFIG, resources)

    print("Parsed Olive config: ", olive_config)

    olive_run(olive_config, setup=True)
    olive_run(olive_config)

    workflow_artifacts_path = Path(workflow_artifacts_args.workflow_artifacts)

    remote_cache_dir = workflow_artifacts_path / current_time / "cache"
    remote_output_dir = workflow_artifacts_path / current_time / "output"

    copy_dir(olive_config["engine"]["cache_dir"], remote_cache_dir)
    copy_dir(olive_config["engine"]["output_dir"], remote_output_dir)


if __name__ == "__main__":
    main()
