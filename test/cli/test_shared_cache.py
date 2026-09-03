# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

import pytest

from olive.cli.shared_cache import SharedCacheCommand


def _parse_shared_cache_args(*args: str):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    SharedCacheCommand.register_subcommand(subparsers)
    return parser.parse_args(["shared-cache", *args])


@pytest.mark.parametrize(
    "account_option,container_option",
    [
        ("--account_name", "--container_name"),
        ("--account", "--container"),
    ],
)
def test_shared_cache_accepts_consistent_and_legacy_option_names(account_option: str, container_option: str):
    args = _parse_shared_cache_args(account_option, "account", container_option, "container")

    assert args.account_name == "account"
    assert args.container_name == "container"
