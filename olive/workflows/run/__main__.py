# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.common.utils import set_tempdir
from olive.workflows import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: Custom Run")
    parser.add_argument(
        "--package-config",
        type=str,
        required=False,
        help=(
            "For advanced users. Path to optional package (json) config file with location "
            "of individual pass module implementation and corresponding dependencies."
            "Configuration might also include user owned/proprietary/private pass implementations."
        ),
    )
    parser.add_argument("--run-config", "--config", type=str, help="Path to json config file", required=True)
    parser.add_argument("--setup", help="Whether run environment setup", action="store_true")
    parser.add_argument("--data-root", "--data_root", help="The data root path for optimization", required=False)
    parser.add_argument("--tempdir", type=str, help="Root directory for tempfile directories and files", required=False)

    args = parser.parse_args()

    set_tempdir(args.tempdir)

    var_args = vars(args)
    del var_args["tempdir"]
    run(**var_args)
