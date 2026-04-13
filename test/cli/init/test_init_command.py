# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import patch


class TestInitCommand:
    def test_register_subcommand(self):
        from argparse import ArgumentParser

        from olive.cli.init import InitCommand

        parser = ArgumentParser()
        sub_parsers = parser.add_subparsers()
        InitCommand.register_subcommand(sub_parsers)

        args = parser.parse_args(["init", "-o", "/tmp/out"])
        assert args.output_path == "/tmp/out"
        assert args.func is InitCommand

    @patch("olive.cli.init.wizard.InitWizard")
    def test_run(self, mock_wizard_cls):
        from argparse import ArgumentParser

        from olive.cli.init import InitCommand

        parser = ArgumentParser()
        sub_parsers = parser.add_subparsers()
        InitCommand.register_subcommand(sub_parsers)

        args = parser.parse_args(["init", "-o", "./my-output"])
        cmd = InitCommand(parser, args, [])
        cmd.run()

        mock_wizard_cls.assert_called_once_with(default_output_path="./my-output")
        mock_wizard_cls.return_value.start.assert_called_once()
