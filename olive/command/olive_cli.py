# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from argparse import ArgumentParser

from olive.command.export_adapters import ExportAdaptersCommand
from olive.command.manage_compute import ManageAMLComputeCommand
from olive.command.run import WorkflowRunCommand


def main(raw_args=None, called_as_console_script: bool = True):
    parser = ArgumentParser(
        "Olive CLI tool", usage="olive-cli" if called_as_console_script else "python -m olive.command"
    )
    commands_parser = parser.add_subparsers()

    # Register commands
    WorkflowRunCommand.register_subcommand(commands_parser)
    ManageAMLComputeCommand.register_subcommand(commands_parser)
    ExportAdaptersCommand.register_subcommand(commands_parser)

    args = parser.parse_args(raw_args)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Run the command
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main(called_as_console_script=False)
