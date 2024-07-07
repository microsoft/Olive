# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path
import shutil
from olive.common.constants import WORKFLOW_ARTIFACTS, WORKFLOW_CONFIG
from olive.workflows import run as olive_run


from olive.common.utils import aml_runner_hf_login, copy_dir
from olive.logging import set_verbosity_from_env

def main(raw_args=None):
    set_verbosity_from_env()

    # login to hf if HF_LOGIN is set to True
    aml_runner_hf_login()
    
    parser = argparse.ArgumentParser("Olive workflow config")

    parser.add_argument(f"--{WORKFLOW_CONFIG}", type=str, help="olive workflow config", required=True)
    parser.add_argument(f"--{WORKFLOW_ARTIFACTS}", type=str, help="olive workflow artifacts", required=True)

    olive_config_args = parser.parse_args(raw_args)
    
    with open(olive_config_args.workflow_config) as f:
        olive_config = json.load(f)

    olive_run(olive_config, setup = True)
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
