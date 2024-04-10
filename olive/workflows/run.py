# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from warnings import warn

from olive.command.olive_cli import main as cli_main

if __name__ == "__main__":
    warn(
        "Running `python -m olive.workflows.run` is deprecated and might be removed in the future. Please use"
        " `olive-cli run` or `python -m olive.command run` instead.",
        DeprecationWarning,
    )

    # get args from command line
    args = ["run"] + sys.argv[1:]
    cli_main(args)
