# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import tempfile
from pathlib import Path

from olive.workflows import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: Custom Run")
    parser.add_argument("--config", type=str, help="Path to json config file", required=True)
    parser.add_argument("--setup", help="Whether run environment setup", action="store_true")
    parser.add_argument("--data_root", help="The data root path for optimization", required=False)
    parser.add_argument("--tempdir", type=str, help="Root directory for tempfile directories and files", required=False)

    args = parser.parse_args()

    if args.tempdir is not None:
        tempdir = Path(args.tempdir).resolve()
        tempdir.mkdir(parents=True, exist_ok=True)
        # setting as string to be safe
        tempfile.tempdir = str(tempdir)

    var_args = vars(args)
    del var_args["tempdir"]
    run(**var_args)
