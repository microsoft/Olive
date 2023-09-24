# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.common.utils import get_attr


def test_attr_exists():
    class A:
        def __init__(self, b):
            self.b = b

    class B:
        def __init__(self, c):
            self.c = c

    class C:
        def __init__(self):
            self.d = "hi"

    c = C()
    b = B(c)
    a = A(b)

    attrs = ["", "b", "b.c", "b.c.d"]
    expected = [a, b, c, "hi"]
    for attr, exp in zip(attrs, expected):
        assert get_attr(a, attr) == exp


def test_attr_no_exists():
    a = "hi"

    assert get_attr(a, "b") is None


def test_attr_no_exists_raise():
    a = "hi"

    with pytest.raises(AttributeError):
        get_attr(a, "b", fail_on_not_found=True)
