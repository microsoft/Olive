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
        "--optimize",
        action="store_true",
        help="If set, run transformers optimization pass",
    )
    args = parser.parse_args()

    input_filename = "bert_cuda_gpu.template.json"
    with Path(input_filename).open("r") as f:
        config = json.load(f)

    if not args.optimize:
        del config["passes"]["transformers_optimization"]

    output_filename = input_filename.replace(".template", "")
    with Path(output_filename).open("w") as strm:
        json.dump(config, fp=strm, indent=4)
