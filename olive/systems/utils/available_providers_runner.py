# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# NOTE: Only onnxruntime and its dependencies can be imported in this file.
import argparse
import json
import logging
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Get available execution providers")
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)
    available_eps = []
    try:
        import onnxruntime_qnn as qnn_ep

        ep_lib_path = qnn_ep.get_library_path()
        ep_registration_name = "QNNExecutionProvider"
        ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

        # get available providers for ABI EP with ort 1.24 is broken. Hence the below hack
        available_eps.append("QNNExecutionProvider")
        ort.unregister_execution_provider_library(ep_registration_name)
    except Exception as e:
        logger.warning("Failed to register QNNExecutionProvider: %s", str(e))
    # get available execution providers
    # python environment system doesn't use EP registration yet
    available_eps.extend(ort.get_available_providers())

    # save to json
    with Path(args.output_path).open("w") as f:
        json.dump(available_eps, f)


if __name__ == "__main__":
    main()
