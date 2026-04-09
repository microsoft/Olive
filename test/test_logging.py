# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.logging import (
    enable_filelog,
    get_logger_level,
    get_olive_logger,
    get_verbosity,
    set_default_logger_severity,
    set_verbosity,
    set_verbosity_critical,
    set_verbosity_debug,
    set_verbosity_error,
    set_verbosity_from_env,
    set_verbosity_info,
    set_verbosity_warning,
)


class TestGetOliveLogger:
    def test_returns_olive_logger(self):
        logger = get_olive_logger()
        assert logger.name == "olive"

    def test_returns_same_logger_instance(self):
        logger1 = get_olive_logger()
        logger2 = get_olive_logger()
        assert logger1 is logger2


class TestSetVerbosity:
    def test_set_verbosity_info(self):
        set_verbosity_info()
        assert get_olive_logger().level == logging.INFO

    def test_set_verbosity_warning(self):
        set_verbosity_warning()
        assert get_olive_logger().level == logging.WARNING

    def test_set_verbosity_debug(self):
        set_verbosity_debug()
        assert get_olive_logger().level == logging.DEBUG

    def test_set_verbosity_error(self):
        set_verbosity_error()
        assert get_olive_logger().level == logging.ERROR

    def test_set_verbosity_critical(self):
        set_verbosity_critical()
        assert get_olive_logger().level == logging.CRITICAL

    def test_set_verbosity_custom_level(self):
        set_verbosity(logging.WARNING)
        assert get_olive_logger().level == logging.WARNING


class TestSetVerbosityFromEnv:
    def test_set_verbosity_from_env_default(self):
        with patch.dict("os.environ", {}, clear=True):
            set_verbosity_from_env()

    def test_set_verbosity_from_env_custom(self):
        with patch.dict("os.environ", {"OLIVE_LOG_LEVEL": "DEBUG"}):
            set_verbosity_from_env()
            assert get_olive_logger().level == logging.DEBUG


class TestGetVerbosity:
    def test_get_verbosity_returns_int(self):
        set_verbosity_info()
        level = get_verbosity()
        assert isinstance(level, int)
        assert level == logging.INFO


class TestGetLoggerLevel:
    @pytest.mark.parametrize(
        ("level_int", "expected"),
        [
            (0, logging.DEBUG),
            (1, logging.INFO),
            (2, logging.WARNING),
            (3, logging.ERROR),
            (4, logging.CRITICAL),
        ],
    )
    def test_valid_levels(self, level_int, expected):
        assert get_logger_level(level_int) == expected

    @pytest.mark.parametrize("invalid_level", [-1, 5, 10, 100])
    def test_invalid_levels_raise_value_error(self, invalid_level):
        with pytest.raises(ValueError, match="Invalid level"):
            get_logger_level(invalid_level)


class TestSetDefaultLoggerSeverity:
    @pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
    def test_set_default_logger_severity(self, level):
        set_default_logger_severity(level)
        expected = get_logger_level(level)
        assert get_olive_logger().level == expected


class TestEnableFilelog:
    def test_enable_filelog_creates_handler(self, tmp_path):
        workflow_id = "test_workflow"
        enable_filelog(1, str(tmp_path), workflow_id)

        logger = get_olive_logger()
        log_file_path = tmp_path / f"{workflow_id}.log"

        # Check that a file handler was added
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0

        # Clean up: remove the handler we added
        for h in file_handlers:
            if Path(h.baseFilename) == log_file_path.resolve():
                logger.removeHandler(h)
                h.close()
