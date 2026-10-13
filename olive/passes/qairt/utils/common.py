# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from olive.common.utils import StrEnumBase


class QairtLogLevel(StrEnumBase):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"

    def to_genie_log_level(self) -> str:
        """Map to the closest Genie log level string."""
        return {
            QairtLogLevel.ERROR: "error",
            QairtLogLevel.WARNING: "warn",
            QairtLogLevel.INFO: "info",
            QairtLogLevel.DEBUG: "verbose",
            QairtLogLevel.TRACE: "verbose",
        }[self]
