# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from argparse import ArgumentParser
from warnings import warn

from olive.command.configure_qualcomm_sdk import ConfigureQualcommSDKCommand
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
    ConfigureQualcommSDKCommand.register_subcommand(commands_parser)
    ManageAMLComputeCommand.register_subcommand(commands_parser)
    ExportAdaptersCommand.register_subcommand(commands_parser)

    args = parser.parse_args(raw_args)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Run the command
    service = args.func(args)
    service.run()


def legacy_call(deprecated_module: str, command_name: str):
    """Run a command with a warning about the deprecation of the module.

    Command arguments are taken from the command line.

    :param deprecated_module: The deprecated module name.
    :param command_name: The command name to run.
    """
    warn(
        f"Running `python -m {deprecated_module}` is deprecated and might be removed in the future. Please use"
        f" `olive-cli {command_name}` or `python -m olive.command {command_name}` instead.",
        FutureWarning,
    )

    # get args from command line
    raw_args = [command_name] + sys.argv[1:]
    main(raw_args)


if __name__ == "__main__":
    main(called_as_console_script=False)
