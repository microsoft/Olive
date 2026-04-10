# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.exception import EXCEPTIONS_TO_RAISE, OliveError, OliveEvaluationError, OlivePassError


class TestOliveError:
    def test_olive_error_is_exception(self):
        # setup

        # execute
        result = issubclass(OliveError, Exception)

        # assert
        assert result

    def test_olive_error_can_be_raised(self):
        # setup

        # execute & assert
        with pytest.raises(OliveError, match="test error"):
            raise OliveError("test error")

    def test_olive_error_empty_message(self):
        # setup

        # execute & assert
        with pytest.raises(OliveError):
            raise OliveError


class TestOlivePassError:
    def test_olive_pass_error_inherits_olive_error(self):
        # setup

        # execute
        result = issubclass(OlivePassError, OliveError)

        # assert
        assert result

    def test_olive_pass_error_can_be_raised(self):
        # setup

        # execute & assert
        with pytest.raises(OlivePassError, match="pass failed"):
            raise OlivePassError("pass failed")

    def test_olive_pass_error_caught_as_olive_error(self):
        # setup

        # execute & assert
        with pytest.raises(OliveError):
            raise OlivePassError("pass failed")


class TestOliveEvaluationError:
    def test_olive_evaluation_error_inherits_olive_error(self):
        # setup

        # execute
        result = issubclass(OliveEvaluationError, OliveError)

        # assert
        assert result

    def test_olive_evaluation_error_can_be_raised(self):
        # setup

        # execute & assert
        with pytest.raises(OliveEvaluationError, match="evaluation failed"):
            raise OliveEvaluationError("evaluation failed")

    def test_olive_evaluation_error_caught_as_olive_error(self):
        # setup

        # execute & assert
        with pytest.raises(OliveError):
            raise OliveEvaluationError("evaluation failed")


class TestExceptionsToRaise:
    def test_exceptions_to_raise_is_tuple(self):
        # setup

        # execute
        result = isinstance(EXCEPTIONS_TO_RAISE, tuple)

        # assert
        assert result

    def test_exceptions_to_raise_contains_expected_types(self):
        # setup
        expected = {AssertionError, AttributeError, ImportError, TypeError, ValueError}

        # execute
        result = set(EXCEPTIONS_TO_RAISE)

        # assert
        assert result == expected

    @pytest.mark.parametrize("exc_type", EXCEPTIONS_TO_RAISE)
    def test_each_exception_is_catchable(self, exc_type):
        # setup

        # execute & assert
        with pytest.raises(exc_type):
            raise exc_type("test")
