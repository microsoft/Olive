# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from olive.command.base import BaseOliveCLICommand
from olive.common.utils import set_tempdir
from olive.workflows import add_run_args, run


class WorkflowRunCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):

        run_parser = parser.add_parser("run", help="Run an olive workflow")
        add_run_args(run_parser)
        run_parser.set_defaults(func=WorkflowRunCommand)

    def run(self):
        set_tempdir(self.args.tempdir)

        var_args = vars(self.args)
        del var_args["func"], var_args["tempdir"]
        run(**var_args)
