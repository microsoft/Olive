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
        # execute
        logger = get_olive_logger()

        # assert
        assert logger.name == "olive"

    def test_returns_same_logger_instance(self):
        # execute
        logger1 = get_olive_logger()
        logger2 = get_olive_logger()

        # assert
        assert logger1 is logger2


class TestSetVerbosity:
    def test_set_verbosity_info(self):
        # execute
        set_verbosity_info()

        # assert
        assert get_olive_logger().level == logging.INFO

    def test_set_verbosity_warning(self):
        # execute
        set_verbosity_warning()

        # assert
        assert get_olive_logger().level == logging.WARNING

    def test_set_verbosity_debug(self):
        # execute
        set_verbosity_debug()

        # assert
        assert get_olive_logger().level == logging.DEBUG

    def test_set_verbosity_error(self):
        # execute
        set_verbosity_error()

        # assert
        assert get_olive_logger().level == logging.ERROR

    def test_set_verbosity_critical(self):
        # execute
        set_verbosity_critical()

        # assert
        assert get_olive_logger().level == logging.CRITICAL

    def test_set_verbosity_custom_level(self):
        # execute
        set_verbosity(logging.WARNING)

        # assert
        assert get_olive_logger().level == logging.WARNING


class TestSetVerbosityFromEnv:
    def test_set_verbosity_from_env_default(self):
        # execute
        with patch.dict("os.environ", {}, clear=True):
            set_verbosity_from_env()

        # assert (no exception raised)

    def test_set_verbosity_from_env_custom(self):
        # execute
        with patch.dict("os.environ", {"OLIVE_LOG_LEVEL": "DEBUG"}):
            set_verbosity_from_env()

            # assert
            assert get_olive_logger().level == logging.DEBUG


class TestGetVerbosity:
    def test_get_verbosity_returns_int(self):
        # setup
        set_verbosity_info()

        # execute
        level = get_verbosity()

        # assert
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
        # execute
        result = get_logger_level(level_int)

        # assert
        assert result == expected

    @pytest.mark.parametrize("invalid_level", [-1, 5, 10, 100])
    def test_invalid_levels_raise_value_error(self, invalid_level):
        # execute & assert
        with pytest.raises(ValueError, match="Invalid level"):
            get_logger_level(invalid_level)


class TestSetDefaultLoggerSeverity:
    @pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
    def test_set_default_logger_severity(self, level):
        # setup
        expected = get_logger_level(level)

        # execute
        set_default_logger_severity(level)

        # assert
        assert get_olive_logger().level == expected


class TestEnableFilelog:
    def test_enable_filelog_creates_handler(self, tmp_path):
        # setup
        workflow_id = "test_workflow"

        # execute
        enable_filelog(1, str(tmp_path), workflow_id)

        # assert
        logger = get_olive_logger()
        log_file_path = tmp_path / f"{workflow_id}.log"
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0

        # cleanup
        for h in file_handlers:
            if Path(h.baseFilename) == log_file_path.resolve():
                logger.removeHandler(h)
                h.close()
