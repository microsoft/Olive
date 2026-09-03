# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from unittest.mock import patch

import pytest

from olive.cli.run_pass import RunPassCommand


def test_list_passes_failure_exits_nonzero():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    RunPassCommand.register_subcommand(sub_parsers)
    args = parser.parse_args(["run-pass", "--list-passes"])
    command = RunPassCommand(parser, args)

    with patch("olive.package_config.OlivePackageConfig.load_default_config", side_effect=RuntimeError("broken config")):
        with pytest.raises(SystemExit) as exc_info:
            command._list_passes()

    assert exc_info.value.code == 1
