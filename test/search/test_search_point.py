# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.search.search_parameter import Categorical, SpecialParamValue
from olive.search.search_space import SearchSpace


class TestSearchPoint:
    @pytest.mark.parametrize("depth", [0, 1, 2])
    def test_is_valid_rejects_invalid_choices(self, depth):
        search_space = SearchSpace([("choice", Categorical([SpecialParamValue.INVALID]))])
        for _ in range(depth):
            search_space = SearchSpace([("nested", Categorical([search_space]))])

        assert not search_space[0].is_valid()

    def test_is_valid_accepts_ignored_and_falsy_choices(self):
        search_space = SearchSpace([("choice", Categorical([SpecialParamValue.IGNORED, None, False, 0, "valid"]))])

        assert all(point.is_valid() for point in search_space)
