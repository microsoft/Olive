# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.exception import EXCEPTIONS_TO_RAISE, OliveError, OliveEvaluationError, OlivePassError


class TestOliveError:
    def test_olive_error_is_exception(self):
        assert issubclass(OliveError, Exception)

    def test_olive_error_can_be_raised(self):
        with pytest.raises(OliveError, match="test error"):
            raise OliveError("test error")

    def test_olive_error_empty_message(self):
        with pytest.raises(OliveError):
            raise OliveError


class TestOlivePassError:
    def test_olive_pass_error_inherits_olive_error(self):
        assert issubclass(OlivePassError, OliveError)

    def test_olive_pass_error_can_be_raised(self):
        with pytest.raises(OlivePassError, match="pass failed"):
            raise OlivePassError("pass failed")

    def test_olive_pass_error_caught_as_olive_error(self):
        with pytest.raises(OliveError):
            raise OlivePassError("pass failed")


class TestOliveEvaluationError:
    def test_olive_evaluation_error_inherits_olive_error(self):
        assert issubclass(OliveEvaluationError, OliveError)

    def test_olive_evaluation_error_can_be_raised(self):
        with pytest.raises(OliveEvaluationError, match="evaluation failed"):
            raise OliveEvaluationError("evaluation failed")

    def test_olive_evaluation_error_caught_as_olive_error(self):
        with pytest.raises(OliveError):
            raise OliveEvaluationError("evaluation failed")


class TestExceptionsToRaise:
    def test_exceptions_to_raise_is_tuple(self):
        assert isinstance(EXCEPTIONS_TO_RAISE, tuple)

    def test_exceptions_to_raise_contains_expected_types(self):
        expected = {AssertionError, AttributeError, ImportError, TypeError, ValueError}
        assert set(EXCEPTIONS_TO_RAISE) == expected

    @pytest.mark.parametrize("exc_type", EXCEPTIONS_TO_RAISE)
    def test_each_exception_is_catchable(self, exc_type):
        with pytest.raises(exc_type):
            raise exc_type("test")
