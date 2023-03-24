# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.workflows import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: Custom Run")
    parser.add_argument("--config", type=str, help="Path to json config file", required=True)

    args = parser.parse_args()

    run(**vars(args))
