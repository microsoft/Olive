# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.common.utils import set_tempdir
from olive.workflows import add_run_args, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: Custom Run")
    add_run_args(parser)

    args = parser.parse_args()

    set_tempdir(args.tempdir)

    var_args = vars(args)
    del var_args["tempdir"]
    run(**var_args)
