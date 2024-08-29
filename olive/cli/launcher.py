# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from argparse import ArgumentParser
from warnings import warn

from olive.cli.auto_opt import AutoOptCommand
from olive.cli.capture_onnx import CaptureOnnxGraphCommand
from olive.cli.cloud_cache import CloudCacheCommand
from olive.cli.configure_qualcomm_sdk import ConfigureQualcommSDKCommand
from olive.cli.export_adapters import ExportAdaptersCommand
from olive.cli.finetune import FineTuneCommand
from olive.cli.manage_aml_compute import ManageAMLComputeCommand
from olive.cli.perf_tuning import PerfTuningCommand
from olive.cli.run import WorkflowRunCommand


def get_cli_parser(called_as_console_script: bool = True) -> ArgumentParser:
    """Get the CLI parser for Olive.

    :param called_as_console_script: Whether the script was called as a console script.
    :return: The CLI parser.
    """
    parser = ArgumentParser("Olive CLI tool", usage="olive" if called_as_console_script else "python -m olive")
    commands_parser = parser.add_subparsers()

    # Register commands
    AutoOptCommand.register_subcommand(commands_parser)
    CaptureOnnxGraphCommand.register_subcommand(commands_parser)
    WorkflowRunCommand.register_subcommand(commands_parser)
    FineTuneCommand.register_subcommand(commands_parser)
    ExportAdaptersCommand.register_subcommand(commands_parser)
    ConfigureQualcommSDKCommand.register_subcommand(commands_parser)
    ManageAMLComputeCommand.register_subcommand(commands_parser)
    PerfTuningCommand.register_subcommand(commands_parser)
    CloudCacheCommand.register_subcommand(commands_parser)

    return parser


def main(raw_args=None, called_as_console_script: bool = True):
    parser = get_cli_parser(called_as_console_script)

    args, unknown_args = parser.parse_known_args(raw_args)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    # Run the command
    service = args.func(parser, args, unknown_args)
    service.run()


def legacy_call(deprecated_module: str, command_name: str, *args):
    """Run a command with a warning about the deprecation of the module.

    Command arguments are taken from the command line.

    :param deprecated_module: The deprecated module name.
    :param command_name: The command name to run.
    :param args: Additional arguments to pass to the command.
    """
    warn(
        f"Running `python -m {deprecated_module}` is deprecated and might be removed in the future. Please use"
        f" `olive {command_name}` or `python -m olive {command_name}` instead.",
        FutureWarning,
    )

    # get args from command line
    raw_args = [command_name, *args]
    main(raw_args)


if __name__ == "__main__":
    main(called_as_console_script=False)
