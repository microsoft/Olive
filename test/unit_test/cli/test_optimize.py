# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

import pytest

from olive.cli.optimize import OptimizeCommand


class TestOptimizeCommand:
    def test_register_subcommand(self):
        """Test that the optimize command registers properly."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        # This should not raise any exception
        OptimizeCommand.register_subcommand(subparsers)

        # Parse help to ensure command is registered
        with pytest.raises(SystemExit):  # argparse raises SystemExit on --help
            parser.parse_args(["optimize", "--help"])
