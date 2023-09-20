# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

import onnxruntime as ort


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Get available execution providers")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path of directory to save the available eps json."
    )

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # get available execution providers
    available_eps = ort.get_available_providers()

    # save available execution providers to file
    output_json_path = Path(args.output_path) / "available_eps.json"
    json.dump(available_eps, open(output_json_path, "w"), indent=4)


if __name__ == "__main__":
    main()
