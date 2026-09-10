# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest


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
        telemetry = MagicMock(accepts_detailed_events=True)
        with (
            patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
            patch("olive.telemetry.telemetry_extensions._resolve_invoked_from", return_value="test"),
            patch("olive.telemetry.telemetry_extensions.log_action") as log_action,
            patch("olive.telemetry.telemetry_extensions.log_error") as log_error,
        ):
            cmd.run()

        mock_wizard_cls.assert_called_once_with(default_output_path="./my-output")
        mock_wizard_cls.return_value.start.assert_called_once()
        log_action.assert_called_once()
        assert log_action.call_args.kwargs["action_name"] == "Init"
        assert log_action.call_args.kwargs["success"] is True
        log_error.assert_not_called()

    @patch("olive.cli.init.wizard.InitWizard")
    def test_run_failure_emits_one_action_and_error(self, mock_wizard_cls):
        from argparse import ArgumentParser

        from olive.cli.init import InitCommand

        parser = ArgumentParser()
        sub_parsers = parser.add_subparsers()
        InitCommand.register_subcommand(sub_parsers)
        args = parser.parse_args(["init"])
        cmd = InitCommand(parser, args, [])
        mock_wizard_cls.return_value.start.side_effect = RuntimeError("boom")

        telemetry = MagicMock(accepts_detailed_events=True)
        with (
            patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
            patch("olive.telemetry.telemetry_extensions._resolve_invoked_from", return_value="test"),
            patch("olive.telemetry.telemetry_extensions.log_action") as log_action,
            patch("olive.telemetry.telemetry_extensions.log_error") as log_error,
            pytest.raises(RuntimeError, match="boom"),
        ):
            cmd.run()

        log_action.assert_called_once()
        assert log_action.call_args.kwargs["action_name"] == "Init"
        assert log_action.call_args.kwargs["success"] is False
        log_error.assert_called_once()
