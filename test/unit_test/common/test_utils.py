# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform

import pytest

from olive.common.constants import OS
from olive.common.utils import get_path_by_os


@pytest.mark.parametrize(
    ("path", "target_system", "expected"),
    [
        ("ucm/olive", OS.LINUX, "ucm/olive"),
        ("ucm/olive", OS.WINDOWS, r"ucm\\olive"),
    ],
)
@pytest.mark.skipif(platform.system() != OS.LINUX, reason="Only runs on Linux")
def test_get_path_by_os_linux(path, target_system, expected):
    assert get_path_by_os(path, target_system) == expected


@pytest.mark.parametrize(
    ("path", "target_system", "expected"),
    [
        (r"ucm\\olive", OS.LINUX, "ucm/olive"),
        (r"ucm\\olive", OS.WINDOWS, r"ucm\\olive"),
    ],
)
@pytest.mark.skipif(platform.system() != OS.WINDOWS, reason="Only runs on Windows")
def test_get_path_by_os_windows(path, target_system, expected):
    assert get_path_by_os(path, target_system) == expected
