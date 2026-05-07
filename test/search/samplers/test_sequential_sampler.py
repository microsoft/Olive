# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from unittest.mock import patch

from olive.search.samplers.sequential_sampler import SequentialSampler
from olive.search.search_parameter import Categorical
from olive.search.search_space import SearchSpace


class TestSequentialSampler:
    @patch("olive.search.search_space.SearchSpace.__getitem__")
    def test_length(self, mock_search_space_get_item):
        search_space = SearchSpace(
            [
                ("PassA", Categorical([1, 2])),
                ("PassB", Categorical([1, 2, 3])),
                ("PassC", Categorical(["a", "b"])),
                ("PassD", Categorical(["a", "b", "c"])),
            ]
        )
        assert len(search_space) == 36

        sampler = SequentialSampler(search_space)

        suggestions = []
        while not sampler.should_stop:
            suggestions.append(sampler.suggest())

        assert len(suggestions) == 36
        assert mock_search_space_get_item.call_count == 36

    def test_iteration(self):
        search_space = SearchSpace(
            [
                ("PassA", Categorical([1, 2])),
                ("PassB", Categorical([1, 2, 3])),
                ("PassC", Categorical(["a", "b"])),
                ("PassD", Categorical(["a", "b", "c"])),
            ]
        )
        assert len(search_space) == 36

        sampler = SequentialSampler(search_space)

        actual = []
        while not sampler.should_stop:
            actual.append(sampler.suggest())

        actual = [(search_point.index, search_point.values) for search_point in actual]
        expected = [
            (0, OrderedDict([("PassA", (0, 1)), ("PassB", (0, 1)), ("PassC", (0, "a")), ("PassD", (0, "a"))])),
            (1, OrderedDict([("PassA", (1, 2)), ("PassB", (0, 1)), ("PassC", (0, "a")), ("PassD", (0, "a"))])),
            (2, OrderedDict([("PassA", (0, 1)), ("PassB", (1, 2)), ("PassC", (0, "a")), ("PassD", (0, "a"))])),
            (3, OrderedDict([("PassA", (1, 2)), ("PassB", (1, 2)), ("PassC", (0, "a")), ("PassD", (0, "a"))])),
            (4, OrderedDict([("PassA", (0, 1)), ("PassB", (2, 3)), ("PassC", (0, "a")), ("PassD", (0, "a"))])),
            (5, OrderedDict([("PassA", (1, 2)), ("PassB", (2, 3)), ("PassC", (0, "a")), ("PassD", (0, "a"))])),
            (6, OrderedDict([("PassA", (0, 1)), ("PassB", (0, 1)), ("PassC", (1, "b")), ("PassD", (0, "a"))])),
            (7, OrderedDict([("PassA", (1, 2)), ("PassB", (0, 1)), ("PassC", (1, "b")), ("PassD", (0, "a"))])),
            (8, OrderedDict([("PassA", (0, 1)), ("PassB", (1, 2)), ("PassC", (1, "b")), ("PassD", (0, "a"))])),
            (9, OrderedDict([("PassA", (1, 2)), ("PassB", (1, 2)), ("PassC", (1, "b")), ("PassD", (0, "a"))])),
            (10, OrderedDict([("PassA", (0, 1)), ("PassB", (2, 3)), ("PassC", (1, "b")), ("PassD", (0, "a"))])),
            (11, OrderedDict([("PassA", (1, 2)), ("PassB", (2, 3)), ("PassC", (1, "b")), ("PassD", (0, "a"))])),
            (12, OrderedDict([("PassA", (0, 1)), ("PassB", (0, 1)), ("PassC", (0, "a")), ("PassD", (1, "b"))])),
            (13, OrderedDict([("PassA", (1, 2)), ("PassB", (0, 1)), ("PassC", (0, "a")), ("PassD", (1, "b"))])),
            (14, OrderedDict([("PassA", (0, 1)), ("PassB", (1, 2)), ("PassC", (0, "a")), ("PassD", (1, "b"))])),
            (15, OrderedDict([("PassA", (1, 2)), ("PassB", (1, 2)), ("PassC", (0, "a")), ("PassD", (1, "b"))])),
            (16, OrderedDict([("PassA", (0, 1)), ("PassB", (2, 3)), ("PassC", (0, "a")), ("PassD", (1, "b"))])),
            (17, OrderedDict([("PassA", (1, 2)), ("PassB", (2, 3)), ("PassC", (0, "a")), ("PassD", (1, "b"))])),
            (18, OrderedDict([("PassA", (0, 1)), ("PassB", (0, 1)), ("PassC", (1, "b")), ("PassD", (1, "b"))])),
            (19, OrderedDict([("PassA", (1, 2)), ("PassB", (0, 1)), ("PassC", (1, "b")), ("PassD", (1, "b"))])),
            (20, OrderedDict([("PassA", (0, 1)), ("PassB", (1, 2)), ("PassC", (1, "b")), ("PassD", (1, "b"))])),
            (21, OrderedDict([("PassA", (1, 2)), ("PassB", (1, 2)), ("PassC", (1, "b")), ("PassD", (1, "b"))])),
            (22, OrderedDict([("PassA", (0, 1)), ("PassB", (2, 3)), ("PassC", (1, "b")), ("PassD", (1, "b"))])),
            (23, OrderedDict([("PassA", (1, 2)), ("PassB", (2, 3)), ("PassC", (1, "b")), ("PassD", (1, "b"))])),
            (24, OrderedDict([("PassA", (0, 1)), ("PassB", (0, 1)), ("PassC", (0, "a")), ("PassD", (2, "c"))])),
            (25, OrderedDict([("PassA", (1, 2)), ("PassB", (0, 1)), ("PassC", (0, "a")), ("PassD", (2, "c"))])),
            (26, OrderedDict([("PassA", (0, 1)), ("PassB", (1, 2)), ("PassC", (0, "a")), ("PassD", (2, "c"))])),
            (27, OrderedDict([("PassA", (1, 2)), ("PassB", (1, 2)), ("PassC", (0, "a")), ("PassD", (2, "c"))])),
            (28, OrderedDict([("PassA", (0, 1)), ("PassB", (2, 3)), ("PassC", (0, "a")), ("PassD", (2, "c"))])),
            (29, OrderedDict([("PassA", (1, 2)), ("PassB", (2, 3)), ("PassC", (0, "a")), ("PassD", (2, "c"))])),
            (30, OrderedDict([("PassA", (0, 1)), ("PassB", (0, 1)), ("PassC", (1, "b")), ("PassD", (2, "c"))])),
            (31, OrderedDict([("PassA", (1, 2)), ("PassB", (0, 1)), ("PassC", (1, "b")), ("PassD", (2, "c"))])),
            (32, OrderedDict([("PassA", (0, 1)), ("PassB", (1, 2)), ("PassC", (1, "b")), ("PassD", (2, "c"))])),
            (33, OrderedDict([("PassA", (1, 2)), ("PassB", (1, 2)), ("PassC", (1, "b")), ("PassD", (2, "c"))])),
            (34, OrderedDict([("PassA", (0, 1)), ("PassB", (2, 3)), ("PassC", (1, "b")), ("PassD", (2, "c"))])),
            (35, OrderedDict([("PassA", (1, 2)), ("PassB", (2, 3)), ("PassC", (1, "b")), ("PassD", (2, "c"))])),
        ]
        assert actual == expected
