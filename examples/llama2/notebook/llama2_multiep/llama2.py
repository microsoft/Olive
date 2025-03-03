# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "multi_ep"],
        help="Device to use",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="If set, run transformers optimization pass",
    )
    args = parser.parse_args()

    input_filename = f"config_{args.device}.template.json"
    with Path(input_filename).open("r") as f:
        config = json.load(f)

    if not args.quantize:
        del config["passes"]["blockwise_quant_int4"]

    output_filename = input_filename.replace(".template", "")
    with Path(output_filename).open("w") as strm:
        json.dump(config, fp=strm, indent=4)
