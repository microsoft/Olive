import argparse
import json
import os
import sys

from olive.common.hf.login import huggingface_login
from olive.logging import get_olive_logger, set_verbosity_from_env
from olive.telemetry.telemetry import Telemetry
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
    try:
        olive_run(config, emit_error_telemetry=False, emit_recipe_telemetry=False)
    finally:
        telemetry = Telemetry.get_existing_instance()
        if telemetry is not None:
            telemetry.shutdown(
                timeout_millis=15_000,
                callback_timeout_millis=15_000,
                flush_seconds=15,
            )


if __name__ == "__main__":
    set_verbosity_from_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="runner config")

    args, _ = parser.parse_known_args()
    logger.info("command line arguments: %s", sys.argv)
    runner_entry(args.config)
