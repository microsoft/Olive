# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.common.utils import retry_func

# pylint: disable=global-statement


num_tries = 0


def fail_with_key_error():
    global num_tries
    if num_tries == 0:
        num_tries += 1
        raise KeyError("This is a key error")
    else:
        return True


def return_args(*args, **kwargs):
    return args, kwargs


@pytest.mark.parametrize("exceptions", [KeyError, (KeyError, ValueError), Exception])
def test_success(exceptions):
    global num_tries
    num_tries = 0
    assert retry_func(fail_with_key_error, max_tries=2, delay=1, exceptions=exceptions)
    assert num_tries == 1


def test_failure():
    global num_tries
    num_tries = 0
    with pytest.raises(KeyError):
        retry_func(fail_with_key_error, max_tries=1, delay=1)


def test_args():
    assert retry_func(return_args, [1, 2, 3], {"a": 4, "b": 5}) == ((1, 2, 3), {"a": 4, "b": 5})


def test_different_exceptions():
    global num_tries
    num_tries = 0
    with pytest.raises(KeyError):
        retry_func(fail_with_key_error, max_tries=2, delay=1, exceptions=ValueError)
