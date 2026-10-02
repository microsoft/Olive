# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect

from olive.cli.run_pass import RunPassCommand


def test_run_pass_action_decorates_run_method_not_command_class():
    assert inspect.isclass(RunPassCommand)
    assert hasattr(RunPassCommand.run, "__wrapped__")
