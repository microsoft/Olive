# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.workflows.convertquantize.convertquantize import convertquantize

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: ConvertQuantize")
    parser.add_argument("--config", type=str, help="Path to json config file", required=True)

    args = parser.parse_args()

    convertquantize(**vars(args))
