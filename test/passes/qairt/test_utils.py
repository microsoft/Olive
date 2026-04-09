# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import pytest

from olive.passes.qairt.utils import QairtLogLevel


@pytest.mark.parametrize("value", ["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"])
def test_qairt_log_level_string_equality(value):
    """QairtLogLevel members compare equal to their string values."""
    assert QairtLogLevel(value) == value


@pytest.mark.parametrize(
    ("log_level", "expected"),
    [
        (QairtLogLevel.ERROR, "error"),
        (QairtLogLevel.WARNING, "warn"),
        (QairtLogLevel.INFO, "info"),
        (QairtLogLevel.DEBUG, "verbose"),
        (QairtLogLevel.TRACE, "verbose"),
    ],
)
def test_to_genie_log_level(log_level, expected):
    """Each QairtLogLevel maps to the correct Genie log level string."""
    assert log_level.to_genie_log_level() == expected


def test_qairt_log_level_invalid_value():
    """Invalid values are rejected by the enum."""
    with pytest.raises(ValueError, match="'WARN' is not a valid QairtLogLevel"):
        QairtLogLevel("WARN")
