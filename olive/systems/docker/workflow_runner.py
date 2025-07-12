import argparse
import json
import os
import sys

from olive.common.hf.login import huggingface_login
from olive.logging import get_olive_logger, set_verbosity_from_env
from olive.workflows import run as olive_run

logger = get_olive_logger()


def runner_entry(config):
    set_verbosity_from_env()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_login(hf_token)

    with open(config) as f:
        config = json.load(f)

    logger.info("Running workflow with config: %s", config)
    olive_run(config, setup=True)
    olive_run(config)


if __name__ == "__main__":
    set_verbosity_from_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="runner config")

    args, _ = parser.parse_known_args()
    logger.info("command line arguments: %s", sys.argv)
    runner_entry(args.config)
