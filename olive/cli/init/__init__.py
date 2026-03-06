# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

from olive.cli.base import BaseOliveCLICommand


class InitCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "init",
            help="Interactive wizard to configure and generate Olive optimization commands.",
        )
        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            default="./olive-output",
            help="Default output directory for the generated command. Default is ./olive-output.",
        )
        sub_parser.set_defaults(func=InitCommand)

    def run(self):
        from olive.cli.init.wizard import InitWizard

        wizard = InitWizard(default_output_path=self.args.output_path)
        wizard.start()
