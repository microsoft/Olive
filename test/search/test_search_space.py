# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict

from olive.search.search_parameter import Categorical, Conditional, SpecialParamValue
from olive.search.search_point import SearchPoint
from olive.search.search_space import SearchSpace

# ruff: noqa: PD011


class TestSearchSpace:
    def test_length_categoricals(self):
        search_space = SearchSpace(
            [
                ("Pass/0/SP1", Categorical([1, 2])),
                ("Pass/0/SP2", Categorical([1, 2, 3])),
                ("Pass/0/SP3", Categorical([1, 2, 3, 4])),
                ("Pass/0/SP4", Categorical([1, 2, 3, 4, 5])),
                ("Pass/0/SP5", Categorical(["a", "b"])),
                ("Pass/0/SP6", Categorical(["a", "b", "c"])),
                ("Pass/0/SP7", Categorical(["a", "b", "c", "d"])),
                ("Pass/0/SP8", Categorical(["a", "b", "c", "d", "e"])),
            ]
        )
        assert len(search_space) == 14400

    def test_length_with_conditionals(self):
        search_space = SearchSpace(
            [
                ("Pass/0/SP1", Categorical([1, 2, 3])),
                ("Pass/0/SP2", Categorical(["x", "y", "z"])),
                (
                    "Pass/0/SP3",
                    Conditional(
                        parents=("Pass/0/SP1",),
                        support={(1,): Categorical(["a"]), (3,): Categorical(["a", "b", "c"])},
                        default=Conditional.get_ignored_choice(),
                    ),
                ),
                ("Pass/0/SP4", Categorical(["4a", "4b", "4c"])),
                (
                    "Pass/0/SP5",
                    Conditional(
                        parents=("Pass/0/SP1", "Pass/0/SP2"),
                        support={
                            (1, "x"): Categorical(["1x", "x1"]),
                            (2, "y"): Categorical(["2y", "y2"]),
                            (3, "z"): Categorical(["3z", "z3"]),
                        },
                        default=Conditional.get_invalid_choice(),
                    ),
                ),
                ("Pass/0/SP6", Categorical(["5x", "5y", "5z"])),
            ]
        )
        assert len(search_space) == 486

    def test_length_with_search_spaces(self):
        search_space = SearchSpace(
            [
                (
                    "Pass/0",
                    SearchSpace(
                        [
                            ("Pass/0/SP1", Categorical([1, 2])),
                            ("Pass/0/SP2", Categorical([1, 2, 3])),
                        ]
                    ),
                ),
                (
                    "Pass/1",
                    SearchSpace(
                        [
                            ("Pass/1/SP1", Categorical(["a", "b"])),
                            ("Pass/1/SP2", Categorical(["a", "b", "c"])),
                        ]
                    ),
                ),
            ]
        )
        assert len(search_space) == 36

    def test_length(self):
        search_space = SearchSpace(
            [
                (
                    "PassA",
                    Categorical(
                        [SearchSpace([("PA0_SP1", Categorical([1, 2, 3])), ("PA0_SP2", Categorical([4, 5, 6]))])]
                    ),
                ),
                (
                    "PassB",
                    Categorical(
                        [
                            SearchSpace(
                                [("PB0_SP1", Categorical(["a", "b", "c"])), ("PB0_SP2", Categorical(["x", "y", "z"]))]
                            ),
                            SearchSpace(
                                [("PB0_SP1", Categorical(["x", "y", "z"])), ("PB0_SP2", Categorical(["a", "b", "c"]))]
                            ),
                        ]
                    ),
                ),
                ("PassC", SearchSpace([("PC0_SP1", Categorical([3, 2, 1])), ("PC0_SP2", Categorical([9, 8, 7]))])),
                (
                    "PassD",
                    SearchSpace(
                        [
                            ("PD0_SP1", Categorical([1, 2, 3])),
                            ("PD0_SP2", Categorical(["x", "y", "z"])),
                            (
                                "PD0_SP3",
                                Conditional(
                                    parents=("PD0_SP1",),
                                    support={(1,): Categorical(["a"]), (3,): Categorical(["a", "b", "c"])},
                                    default=Conditional.get_ignored_choice(),
                                ),
                            ),
                            (
                                "PD0_SP4",
                                Conditional(
                                    parents=("PD0_SP1", "PD0_SP2"),
                                    support={
                                        (1, "x"): Categorical(["1x", "x1"]),
                                        (2, "y"): Categorical(["2y", "y2"]),
                                        (3, "z"): Categorical(["3y", "y3"]),
                                    },
                                    default=Conditional.get_invalid_choice(),
                                ),
                            ),
                            (
                                "PD0_SP5",
                                Conditional(
                                    parents=("PD0_SP3", "PD0_SP4"),
                                    support={
                                        ("a", "1x"): SearchSpace(
                                            [
                                                ("PD0_SP5_SP1", Categorical(["a1x", "x1a"])),
                                                ("PD0_SP5_SP2", Categorical(["x1a", "a1x"])),
                                            ]
                                        ),
                                        ("c", "3y"): SearchSpace(
                                            [
                                                ("PD0_SP5_SP3", Categorical(["c3y", "y3c"])),
                                                ("PD0_SP5_SP4", Categorical(["y3c", "3cy"])),
                                            ]
                                        ),
                                    },
                                    default=Conditional.get_invalid_choice(),
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )
        # PassA:length = 9
        # PassB:length = 18
        #   PassB[0]:length = 9
        #   PassB[1]:length = 9
        # PassC:length = 9
        # PassD:length = 216
        #   PassD[0].PD0_SP1:length = 3
        #   PassD[0].PD0_SP2:length = 3
        #   PassD[0].PD0_SP3:length = 3
        #   PassD[0].PD0_SP4:length = 2
        #   PassD[0].PD0_SP5:length = 4
        assert len(search_space) == 314928  # 9 * 18 * 9 * 216

    def test_empty(self):
        search_space = SearchSpace([])
        assert len(search_space) == 1

        actual = [(search_point.index, search_point.values) for search_point in search_space]
        expected = [(0, OrderedDict())]
        assert actual == expected

        search_space = SearchSpace(
            [
                ("PassA", Categorical([SearchSpace([]), SearchSpace([]), SearchSpace([])])),
                ("PassB", Categorical([SearchSpace([]), SearchSpace([])])),
                ("PassC", Categorical([SearchSpace([])])),
            ]
        )
        assert len(search_space) == 6

        actual = [(search_point.index, search_point.values) for search_point in search_space]
        expected = [
            (
                0,
                OrderedDict(
                    [("PassA", (0, OrderedDict())), ("PassB", (0, OrderedDict())), ("PassC", (0, OrderedDict()))]
                ),
            ),
            (
                1,
                OrderedDict(
                    [("PassA", (1, OrderedDict())), ("PassB", (0, OrderedDict())), ("PassC", (0, OrderedDict()))]
                ),
            ),
            (
                2,
                OrderedDict(
                    [("PassA", (2, OrderedDict())), ("PassB", (0, OrderedDict())), ("PassC", (0, OrderedDict()))]
                ),
            ),
            (
                3,
                OrderedDict(
                    [("PassA", (0, OrderedDict())), ("PassB", (1, OrderedDict())), ("PassC", (0, OrderedDict()))]
                ),
            ),
            (
                4,
                OrderedDict(
                    [("PassA", (1, OrderedDict())), ("PassB", (1, OrderedDict())), ("PassC", (0, OrderedDict()))]
                ),
            ),
            (
                5,
                OrderedDict(
                    [("PassA", (2, OrderedDict())), ("PassB", (1, OrderedDict())), ("PassC", (0, OrderedDict()))]
                ),
            ),
        ]
        assert actual == expected

    def test_iteration_with_categoricals(self):
        search_space = SearchSpace(
            [
                ("Pass/0/SP1", Categorical([1, 2, 3])),
                ("Pass/0/SP2", Categorical(["a", "b"])),
                ("Pass/0/SP3", Categorical([10, 20, 30])),
                ("Pass/0/SP4", Categorical(["x", "y", "z"])),
            ]
        )

        actual = [(search_point.index, search_point.values) for search_point in search_space]
        expected = [
            (
                0,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                1,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                2,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                3,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                4,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                5,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                6,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                7,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                8,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                9,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                10,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                11,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                12,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                13,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                14,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                15,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                16,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                17,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (0, "x")),
                    ]
                ),
            ),
            (
                18,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                19,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                20,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                21,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                22,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                23,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                24,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                25,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                26,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                27,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                28,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                29,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                30,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                31,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                32,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                33,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                34,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                35,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (1, "y")),
                    ]
                ),
            ),
            (
                36,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                37,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                38,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                39,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                40,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                41,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, 10)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                42,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                43,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                44,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                45,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                46,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                47,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, 20)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                48,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                49,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                50,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                51,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                52,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
            (
                53,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, 30)),
                        ("Pass/0/SP4", (2, "z")),
                    ]
                ),
            ),
        ]
        assert actual == expected

    def test_iteration_with_conditionals(self):
        search_space = SearchSpace(
            [
                ("Pass/0/SP1", Categorical([1, 2, 3])),
                (
                    "Pass/0/SP2",
                    Conditional(
                        parents=("Pass/0/SP1",),
                        support={(1,): Categorical(["a"]), (3,): Categorical(["a", "b", "c"])},
                        default=Conditional.get_ignored_choice(),
                    ),
                ),
                ("Pass/0/SP3", Categorical(["x", "y", "z"])),
                (
                    "Pass/0/SP4",
                    Conditional(
                        parents=("Pass/0/SP1", "Pass/0/SP3"),
                        support={
                            (1, "x"): Categorical(["1x", "x1"]),
                            (2, "y"): Categorical(["2y", "y2"]),
                            (3, "z"): Categorical(["3z", "z3"]),
                        },
                        default=Conditional.get_invalid_choice(),
                    ),
                ),
            ]
        )

        actual = [(search_point.index, search_point.values) for search_point in search_space]
        expected = [
            (
                0,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, "1x")),
                    ]
                ),
            ),
            (
                1,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                2,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                3,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, "1x")),
                    ]
                ),
            ),
            (
                4,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                5,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                6,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, "1x")),
                    ]
                ),
            ),
            (
                7,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                8,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (2, "c")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                9,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                10,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, "2y")),
                    ]
                ),
            ),
            (
                11,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                12,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                13,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, "2y")),
                    ]
                ),
            ),
            (
                14,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                15,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                16,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, "2y")),
                    ]
                ),
            ),
            (
                17,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (2, "c")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                18,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                19,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                20,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, "3z")),
                    ]
                ),
            ),
            (
                21,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                22,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                23,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, "3z")),
                    ]
                ),
            ),
            (
                24,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                25,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                26,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (2, "c")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (0, "3z")),
                    ]
                ),
            ),
            (
                27,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, "x1")),
                    ]
                ),
            ),
            (
                28,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                29,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                30,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, "x1")),
                    ]
                ),
            ),
            (
                31,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                32,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                33,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, "x1")),
                    ]
                ),
            ),
            (
                34,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                35,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (2, "c")),
                        ("Pass/0/SP3", (0, "x")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                36,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                37,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, "y2")),
                    ]
                ),
            ),
            (
                38,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                39,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                40,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, "y2")),
                    ]
                ),
            ),
            (
                41,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                42,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                43,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, "y2")),
                    ]
                ),
            ),
            (
                44,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (2, "c")),
                        ("Pass/0/SP3", (1, "y")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                45,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                46,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (0, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                47,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (0, "a")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, "z3")),
                    ]
                ),
            ),
            (
                48,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                49,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (1, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                50,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (1, "b")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, "z3")),
                    ]
                ),
            ),
            (
                51,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (0, 1)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                52,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (1, 2)),
                        ("Pass/0/SP2", (2, SpecialParamValue.IGNORED)),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, SpecialParamValue.INVALID)),
                    ]
                ),
            ),
            (
                53,
                OrderedDict(
                    [
                        ("Pass/0/SP1", (2, 3)),
                        ("Pass/0/SP2", (2, "c")),
                        ("Pass/0/SP3", (2, "z")),
                        ("Pass/0/SP4", (1, "z3")),
                    ]
                ),
            ),
        ]
        assert actual == expected

    def test_iteration_with_search_spaces(self):
        search_space = SearchSpace(
            [
                (
                    "Pass/0",
                    SearchSpace(
                        [
                            ("Pass/0/SP1", Categorical([1, 2])),
                            ("Pass/0/SP2", Categorical([1, 2, 3])),
                        ]
                    ),
                ),
                (
                    "Pass/1",
                    SearchSpace(
                        [
                            ("Pass/1/SP1", Categorical(["a", "b"])),
                            ("Pass/1/SP2", Categorical(["a", "b", "c"])),
                        ]
                    ),
                ),
            ]
        )
        assert len(search_space) == 36

        actual = [(search_point.index, search_point.values) for search_point in search_space]
        expected = [
            (
                0,
                OrderedDict(
                    [
                        ("Pass/0", (0, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                1,
                OrderedDict(
                    [
                        ("Pass/0", (1, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                2,
                OrderedDict(
                    [
                        ("Pass/0", (2, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                3,
                OrderedDict(
                    [
                        ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                4,
                OrderedDict(
                    [
                        ("Pass/0", (4, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                5,
                OrderedDict(
                    [
                        ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                6,
                OrderedDict(
                    [
                        ("Pass/0", (0, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (1, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                7,
                OrderedDict(
                    [
                        ("Pass/0", (1, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (1, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                8,
                OrderedDict(
                    [
                        ("Pass/0", (2, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (1, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                9,
                OrderedDict(
                    [
                        ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (1, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                10,
                OrderedDict(
                    [
                        ("Pass/0", (4, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (1, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                11,
                OrderedDict(
                    [
                        ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (1, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (0, "a"))]))),
                    ]
                ),
            ),
            (
                12,
                OrderedDict(
                    [
                        ("Pass/0", (0, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (2, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                13,
                OrderedDict(
                    [
                        ("Pass/0", (1, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (2, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                14,
                OrderedDict(
                    [
                        ("Pass/0", (2, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (2, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                15,
                OrderedDict(
                    [
                        ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (2, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                16,
                OrderedDict(
                    [
                        ("Pass/0", (4, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (2, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                17,
                OrderedDict(
                    [
                        ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (2, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                18,
                OrderedDict(
                    [
                        ("Pass/0", (0, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (3, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                19,
                OrderedDict(
                    [
                        ("Pass/0", (1, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (3, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                20,
                OrderedDict(
                    [
                        ("Pass/0", (2, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (3, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                21,
                OrderedDict(
                    [
                        ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (3, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                22,
                OrderedDict(
                    [
                        ("Pass/0", (4, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (3, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                23,
                OrderedDict(
                    [
                        ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (3, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (1, "b"))]))),
                    ]
                ),
            ),
            (
                24,
                OrderedDict(
                    [
                        ("Pass/0", (0, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                25,
                OrderedDict(
                    [
                        ("Pass/0", (1, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                26,
                OrderedDict(
                    [
                        ("Pass/0", (2, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                27,
                OrderedDict(
                    [
                        ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                28,
                OrderedDict(
                    [
                        ("Pass/0", (4, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                29,
                OrderedDict(
                    [
                        ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                30,
                OrderedDict(
                    [
                        ("Pass/0", (0, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (5, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                31,
                OrderedDict(
                    [
                        ("Pass/0", (1, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (0, 1))]))),
                        ("Pass/1", (5, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                32,
                OrderedDict(
                    [
                        ("Pass/0", (2, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (5, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                33,
                OrderedDict(
                    [
                        ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                        ("Pass/1", (5, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                34,
                OrderedDict(
                    [
                        ("Pass/0", (4, OrderedDict([("Pass/0/SP1", (0, 1)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (5, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
            (
                35,
                OrderedDict(
                    [
                        ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                        ("Pass/1", (5, OrderedDict([("Pass/1/SP1", (1, "b")), ("Pass/1/SP2", (2, "c"))]))),
                    ]
                ),
            ),
        ]
        assert actual == expected

    def test_get_item_with_categoricals(self):
        search_space = SearchSpace(
            [
                ("Pass/0/SP1", Categorical([1, 2, 3])),
                ("Pass/0/SP2", Categorical(["a", "b"])),
                ("Pass/0/SP3", Categorical([10, 20, 30])),
                ("Pass/0/SP4", Categorical(["x", "y", "z"])),
            ]
        )

        actual_5 = search_space[5]
        expected_5 = SearchPoint(
            5,
            OrderedDict(
                [
                    ("Pass/0/SP1", (2, 3)),
                    ("Pass/0/SP2", (1, "b")),
                    ("Pass/0/SP3", (0, 10)),
                    ("Pass/0/SP4", (0, "x")),
                ]
            ),
        )
        assert actual_5 == expected_5

        actual_45 = search_space[45]
        expected_45 = SearchPoint(
            45,
            OrderedDict(
                [
                    ("Pass/0/SP1", (0, 1)),
                    ("Pass/0/SP2", (1, "b")),
                    ("Pass/0/SP3", (1, 20)),
                    ("Pass/0/SP4", (2, "z")),
                ]
            ),
        )
        assert actual_45 == expected_45

    def test_get_item_with_conditionals(self):
        search_space = SearchSpace(
            [
                ("Pass/0/SP1", Categorical([1, 2, 3])),
                (
                    "Pass/0/SP2",
                    Conditional(
                        parents=("Pass/0/SP1",),
                        support={(1,): Categorical(["a"]), (3,): Categorical(["a", "b", "c"])},
                        default=Conditional.get_ignored_choice(),
                    ),
                ),
                ("Pass/0/SP3", Categorical(["x", "y", "z"])),
                (
                    "Pass/0/SP4",
                    Conditional(
                        parents=("Pass/0/SP1", "Pass/0/SP3"),
                        support={
                            (1, "x"): Categorical(["1x", "x1"]),
                            (2, "y"): Categorical(["2y", "y2"]),
                            (3, "z"): Categorical(["3z", "z3"]),
                        },
                        default=Conditional.get_invalid_choice(),
                    ),
                ),
            ]
        )

        actual_5 = search_space[5]
        expected_5 = SearchPoint(
            5,
            OrderedDict(
                [
                    ("Pass/0/SP1", (2, 3)),
                    ("Pass/0/SP2", (1, "b")),
                    ("Pass/0/SP3", (0, "x")),
                    ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                ]
            ),
        )
        assert actual_5 == expected_5

        actual_17 = search_space[17]
        expected_17 = SearchPoint(
            17,
            OrderedDict(
                [
                    ("Pass/0/SP1", (2, 3)),
                    ("Pass/0/SP2", (2, "c")),
                    ("Pass/0/SP3", (1, "y")),
                    ("Pass/0/SP4", (0, SpecialParamValue.INVALID)),
                ]
            ),
        )
        assert actual_17 == expected_17

    def test_get_item_with_search_spaces(self):
        search_space = SearchSpace(
            [
                (
                    "Pass/0",
                    SearchSpace(
                        [
                            ("Pass/0/SP1", Categorical([1, 2])),
                            ("Pass/0/SP2", Categorical([1, 2, 3])),
                        ]
                    ),
                ),
                (
                    "Pass/1",
                    SearchSpace(
                        [
                            ("Pass/1/SP1", Categorical(["a", "b"])),
                            ("Pass/1/SP2", Categorical(["a", "b", "c"])),
                        ]
                    ),
                ),
            ]
        )
        assert len(search_space) == 36

        actual_5 = search_space[5]
        expected_5 = SearchPoint(
            5,
            OrderedDict(
                [
                    ("Pass/0", (5, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (2, 3))]))),
                    ("Pass/1", (0, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (0, "a"))]))),
                ]
            ),
        )
        assert actual_5 == expected_5

        actual_27 = search_space[27]
        expected_27 = SearchPoint(
            27,
            OrderedDict(
                [
                    ("Pass/0", (3, OrderedDict([("Pass/0/SP1", (1, 2)), ("Pass/0/SP2", (1, 2))]))),
                    ("Pass/1", (4, OrderedDict([("Pass/1/SP1", (0, "a")), ("Pass/1/SP2", (2, "c"))]))),
                ]
            ),
        )
        assert actual_27 == expected_27
