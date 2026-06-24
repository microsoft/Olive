# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict

from olive.search.search_parameter import SpecialParamValue
from olive.search.search_point import SearchPoint


class TestSearchPoint:
    def _make_point(self, index=0, values=None):
        if values is None:
            values = OrderedDict(
                {
                    "pass1": (
                        0,
                        OrderedDict(
                            {
                                "param1": (0, "a"),
                                "param2": (1, 10),
                            }
                        ),
                    )
                }
            )
        return SearchPoint(index=index, values=values)

    def test_creation(self):
        # execute
        point = self._make_point()

        # assert
        assert point.index == 0

    def test_repr(self):
        # setup
        point = self._make_point()

        # execute
        result = repr(point)

        # assert
        assert "SearchPoint" in result
        assert "0" in result

    def test_equality_same(self):
        # setup
        point1 = self._make_point()
        point2 = self._make_point()

        # execute
        result = point1 == point2

        # assert
        assert result

    def test_equality_different_index(self):
        # setup
        point1 = self._make_point(index=0)
        point2 = self._make_point(index=1)

        # execute
        result = point1 == point2

        # assert
        assert not result

    def test_equality_different_type(self):
        # setup
        point = self._make_point()

        # execute
        result = point == "not a search point"

        # assert
        assert not result

    def test_is_valid_true(self):
        # setup
        point = self._make_point()

        # execute
        result = point.is_valid()

        # assert
        assert result is True

    def test_is_valid_false_with_invalid(self):
        # setup
        values = OrderedDict(
            {
                "pass1": OrderedDict(
                    {
                        "param1": SpecialParamValue.INVALID,
                    }
                )
            }
        )
        point = SearchPoint(index=0, values=values)

        # execute
        result = point.is_valid()

        # assert
        assert result is False

    def test_to_json(self):
        # setup
        point = self._make_point(index=5)

        # execute
        result = point.to_json()

        # assert
        assert result["index"] == 5
        assert "values" in result

    def test_from_json_roundtrip(self):
        # setup
        point = self._make_point(index=3)
        json_data = point.to_json()

        # execute
        restored = SearchPoint.from_json(json_data)

        # assert
        assert restored.index == 3
        assert restored == point
