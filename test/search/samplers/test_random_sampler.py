# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from unittest.mock import patch

from olive.search.samplers.random_sampler import RandomSampler
from olive.search.search_parameter import Categorical
from olive.search.search_space import SearchSpace


class TestRandomSampler:
    @patch("olive.search.search_space.SearchSpace.__getitem__")
    def test_call_count(self, mock_search_space_get_item):
        search_space = SearchSpace(
            [
                ("PassA", Categorical([1, 2])),
                ("PassB", Categorical([1, 2, 3])),
                ("PassC", Categorical(["a", "b"])),
                ("PassD", Categorical(["a", "b", "c"])),
            ]
        )
        assert len(search_space) == 36

        config = {"max_samples": 50}
        sampler = RandomSampler(search_space, config=config)

        count = 0
        while not sampler.should_stop:
            sampler.suggest()
            count += 1

        assert count == 36
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

        config = {"seed": 101, "max_samples": 50}
        sampler = RandomSampler(search_space, config=config)

        count = 0
        actual = []
        while not sampler.should_stop:
            actual.append(sampler.suggest())
            count += 1

        actual = [(search_point.index, search_point.values) for search_point in actual]
        expected = [
            (12, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (1, "b")})),
            (35, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (2, "c")})),
            (23, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (1, "b")})),
            (31, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (2, "c")})),
            (3, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (0, "a")})),
            (24, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (2, "c")})),
            (18, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (1, "b")})),
            (7, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (0, "a")})),
            (25, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (2, "c")})),
            (9, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (0, "a")})),
            (13, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (1, "b")})),
            (21, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (1, "b")})),
            (33, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (2, "c")})),
            (8, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (0, "a")})),
            (16, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (1, "b")})),
            (26, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (2, "c")})),
            (2, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (0, "a")})),
            (15, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (1, "b")})),
            (11, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (0, "a")})),
            (10, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (0, "a")})),
            (4, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (0, "a")})),
            (20, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (1, "b")})),
            (32, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (2, "c")})),
            (27, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (2, "c")})),
            (17, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (1, "b")})),
            (6, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (0, "a")})),
            (29, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (2, "c")})),
            (22, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (1, "b")})),
            (14, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (1, "b")})),
            (30, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (2, "c")})),
            (19, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (1, "b")})),
            (28, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (2, "c")})),
            (1, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (0, "a")})),
            (0, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (0, "a")})),
            (34, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (2, "c")})),
            (5, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (0, "a")})),
        ]

        assert count == 36
        assert actual == expected
        assert len({sp for sp, _ in expected}) == len(expected)
