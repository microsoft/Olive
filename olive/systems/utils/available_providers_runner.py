# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# NOTE: Only onnxruntime and its dependencies can be imported in this file.
import argparse
import json
from pathlib import Path

import onnxruntime as ort


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Get available execution providers")
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # get available execution providers
    available_eps = ort.get_available_providers()

    # save to json
    with Path(args.output_path).open("w") as f:
        json.dump(available_eps, f)


if __name__ == "__main__":
    main()
