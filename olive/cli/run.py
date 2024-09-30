# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from olive.cli.base import BaseOliveCLICommand


class WorkflowRunCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("run", help="Run an olive workflow")
        sub_parser.add_argument("--run-config", "--config", type=str, help="Path to json config file", required=True)
        sub_parser.add_argument("--setup", help="Setup environment needed to run the workflow", action="store_true")
        sub_parser.add_argument("--packages", help="List packages required to run the workflow", action="store_true")
        sub_parser.add_argument(
            "--tempdir", type=str, help="Root directory for tempfile directories and files", required=False
        )
        sub_parser.add_argument(
            "--package-config",
            type=str,
            required=False,
            help=(
                "For advanced users. Path to optional package (json) config file with location "
                "of individual pass module implementation and corresponding dependencies. "
                "Configuration might also include user owned/proprietary/private pass implementations."
            ),
        )
        sub_parser.set_defaults(func=WorkflowRunCommand)

    def run(self):
        from olive.workflows import run as olive_run

        var_args = vars(self.args)
        del var_args["func"]
        olive_run(**var_args)
