# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re
from collections import OrderedDict

import pytest

from olive.evaluator.metric_result import MetricResult
from olive.search.search_parameter import Categorical
from olive.search.search_strategy import SearchStrategy, SearchStrategyConfig, SearchStrategyExecutionOrder

# pylint: disable=protected-access
# ruff: noqa: PD011


class TestSearchStrategy:
    @pytest.mark.parametrize(
        "execution_order", [SearchStrategyExecutionOrder.JOINT, SearchStrategyExecutionOrder.PASS_BY_PASS]
    )
    def test_initialize(self, execution_order, tmpdir):
        config = SearchStrategyConfig(execution_order=execution_order, sampler="random")
        space_config = OrderedDict(
            [
                (
                    "PassA",
                    [
                        {"PA0_SP1": Categorical(["A01a", "A01b"]), "PA0_SP2": Categorical(["A02a", "A02b"])},
                        {"PA1_SP1": Categorical(["A11a", "A11b"])},
                        {},
                        {},
                    ],
                ),
                ("PassB", [{"PB0_SP1": Categorical(["B01a", "B01b"])}, {}]),
                ("PassC", [{}, {}]),
            ]
        )

        strategy = SearchStrategy(config)
        strategy.initialize(space_config, "whatever", {})

        actual = re.sub(r"[\n\t\s]*", "", str(strategy._search_spaces))

        if execution_order == SearchStrategyExecutionOrder.JOINT:
            assert len(strategy._search_spaces) == 1

            expected = re.sub(
                r"[\n\t\s]*",
                "",
                """[
                SearchSpace([
                    ('PassA', Categorical([
                            SearchSpace([
                                ('PA0_SP1', Categorical(['A01a', 'A01b'])),
                                ('PA0_SP2', Categorical(['A02a', 'A02b']))
                            ], 4),
                            SearchSpace([('PA1_SP1', Categorical(['A11a', 'A11b']))], 2),
                            SearchSpace([], 1),
                            SearchSpace([], 1)
                        ])
                    ),
                    ('PassB', Categorical([
                            SearchSpace([('PB0_SP1', Categorical(['B01a', 'B01b']))], 2),
                            SearchSpace([], 1)
                        ])
                    ),
                    ('PassC', Categorical([SearchSpace([], 1), SearchSpace([], 1)]))
                ], 48)
            ]""",
            )

        elif execution_order == SearchStrategyExecutionOrder.PASS_BY_PASS:
            assert len(strategy._search_spaces) == 3

            expected = re.sub(
                r"[\n\t\s]*",
                "",
                """[
                SearchSpace([
                    ('PassA', Categorical([
                            SearchSpace([
                                ('PA0_SP1', Categorical(['A01a', 'A01b'])),
                                ('PA0_SP2', Categorical(['A02a', 'A02b']))
                            ], 4),
                            SearchSpace([('PA1_SP1', Categorical(['A11a', 'A11b']))], 2),
                            SearchSpace([], 1),
                            SearchSpace([], 1)
                        ])
                    )], 8),
                SearchSpace([
                    ('PassB', Categorical([
                            SearchSpace([('PB0_SP1', Categorical(['B01a', 'B01b']))], 2),
                            SearchSpace([], 1)
                        ])
                    )], 3),
                SearchSpace([('PassC', Categorical([SearchSpace([], 1), SearchSpace([], 1)]))], 2)
            ]""",
            )
        else:
            expected = None
            pytest.fail("Unsupported execution_order")

        assert actual == expected

    def test_iteration_joint(self, tmpdir):
        config = SearchStrategyConfig(execution_order="joint", sampler="sequential")
        space_config = OrderedDict(
            [
                (
                    "PassA",
                    [
                        {"PA0_SP1": Categorical(["A01a", "A01b"]), "PA0_SP2": Categorical(["A02a", "A02b"])},
                        {"PA1_SP1": Categorical(["A11a", "A11b"])},
                        {},
                        {},
                    ],
                ),
                ("PassB", [{"PB0_SP1": Categorical(["B01a", "B01b"])}, {}]),
                ("PassC", [{}, {}]),
            ]
        )

        strategy = SearchStrategy(config)
        strategy.initialize(space_config, "whatever", {})

        actual = [(sample.search_point.index, sample.passes_configs) for sample in strategy]
        expected = [
            (
                0,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                1,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                2,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                3,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                4,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11a")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                5,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11b")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                6,
                {
                    "PassA": OrderedDict([("index", 2), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                7,
                {
                    "PassA": OrderedDict([("index", 3), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                8,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                9,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                10,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                11,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                12,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11a")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                13,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11b")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                14,
                {
                    "PassA": OrderedDict([("index", 2), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                15,
                {
                    "PassA": OrderedDict([("index", 3), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                16,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                17,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                18,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                19,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                20,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11a")]))]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                21,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11b")]))]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                22,
                {
                    "PassA": OrderedDict([("index", 2), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                23,
                {
                    "PassA": OrderedDict([("index", 3), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 0), ("params", OrderedDict())]),
                },
            ),
            (
                24,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                25,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                26,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                27,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                28,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11a")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                29,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11b")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                30,
                {
                    "PassA": OrderedDict([("index", 2), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                31,
                {
                    "PassA": OrderedDict([("index", 3), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01a")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                32,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                33,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                34,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                35,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                36,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11a")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                37,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11b")]))]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                38,
                {
                    "PassA": OrderedDict([("index", 2), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                39,
                {
                    "PassA": OrderedDict([("index", 3), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 0), ("params", OrderedDict([("PB0_SP1", "B01b")]))]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                40,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                41,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02a")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                42,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01a"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                43,
                {
                    "PassA": OrderedDict(
                        [("index", 0), ("params", OrderedDict([("PA0_SP1", "A01b"), ("PA0_SP2", "A02b")]))]
                    ),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                44,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11a")]))]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                45,
                {
                    "PassA": OrderedDict([("index", 1), ("params", OrderedDict([("PA1_SP1", "A11b")]))]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                46,
                {
                    "PassA": OrderedDict([("index", 2), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
            (
                47,
                {
                    "PassA": OrderedDict([("index", 3), ("params", OrderedDict())]),
                    "PassB": OrderedDict([("index", 1), ("params", OrderedDict())]),
                    "PassC": OrderedDict([("index", 1), ("params", OrderedDict())]),
                },
            ),
        ]

        assert actual == expected

    def test_iteration_pass_by_pass(self, tmpdir):
        config = SearchStrategyConfig(execution_order="pass-by-pass", sampler="sequential")
        space_config = OrderedDict(
            [
                (
                    "PassA",
                    [
                        {"PA0_SP1": Categorical(["A01a", "A01b"]), "PA0_SP2": Categorical(["A02a", "A02b"])},
                        {"PA1_SP1": Categorical(["A11a", "A11b"])},
                        {},
                        {},
                    ],
                ),
                ("PassB", [{"PB0_SP1": Categorical(["B01a", "B01b"])}, {}]),
                ("PassC", [{}, {}]),
            ]
        )
        objectives = OrderedDict(
            [
                (
                    "PassA",
                    {
                        "accuracy-accuracy_custom": {
                            "goal": 0.70,
                            "higher_is_better": True,
                            "priority": 1,
                        },
                        "latency-avg": {"goal": 24.0, "higher_is_better": False, "priority": 2},
                    },
                ),
                (
                    "PassB",
                    {
                        "accuracy-accuracy_custom": {
                            "goal": 0.80,
                            "higher_is_better": True,
                            "priority": 1,
                        },
                        "latency-avg": {"goal": 45.0, "higher_is_better": False, "priority": 2},
                        "latency-max": {"goal": 72.0, "higher_is_better": False, "priority": 2},
                    },
                ),
                (
                    "PassC",
                    {
                        "accuracy-accuracy_custom": {
                            "goal": 0.90,
                            "higher_is_better": True,
                            "priority": 1,
                        },
                        "latency-avg": {"goal": 14.0, "higher_is_better": False, "priority": 2},
                    },
                ),
            ]
        )

        signal1 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.96,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 4.5, "priority": 2, "higher_is_better": False},
            }
        )
        signal2 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.72,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 55.9, "priority": 2, "higher_is_better": False},
            }
        )
        signal3 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.73,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 8.9, "priority": 2, "higher_is_better": False},
            }
        )
        signal4 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.91,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 13.9, "priority": 2, "higher_is_better": False},
            }
        )
        signal5 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.82,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 53.0, "priority": 2, "higher_is_better": False},
                "latency-max": {"value": 60.1, "priority": 2, "higher_is_better": False},
            }
        )
        signal6 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.76,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 58.0, "priority": 2, "higher_is_better": False},
                "latency-max": {"value": 55.2, "priority": 3, "higher_is_better": False},
            }
        )

        signals = {
            "PassA": [
                None,
                (
                    signal1,
                    ["model_id_1"],
                    False,
                ),
                None,
                None,
                (
                    signal2,
                    ["model_id_2"],
                    False,
                ),
                (
                    signal3,
                    ["model_id_3"],
                    True,
                ),
            ],
            "PassB": [
                None,
                None,
                (
                    signal5,
                    ["model_id_5"],
                    False,
                ),
                None,
                None,
                (
                    signal6,
                    ["model_id_6"],
                    False,
                ),
            ],
            "PassC": [
                None,
                None,
                None,
                (
                    signal4,
                    ["model_id_4"],
                    False,
                ),
            ],
        }

        strategy = SearchStrategy(config)
        strategy.initialize(space_config, "whatever", objectives)

        actual = []
        for sample in strategy:
            actual.append((sample.search_point.index, sample.passes_configs, sample.model_ids))

            pass_name = next(iter(sample.search_point.values.keys()))
            if sample.search_point.index < len(signals[pass_name]) and signals[pass_name][sample.search_point.index]:
                strategy.record_feedback_signal(
                    sample.search_point.index, *signals[pass_name][sample.search_point.index]
                )

        expected = [
            (
                0,
                OrderedDict(
                    {"PassA": OrderedDict({"index": 0, "params": OrderedDict({"PA0_SP1": "A01a", "PA0_SP2": "A02a"})})}
                ),
                ["whatever"],
            ),
            (
                1,
                OrderedDict(
                    {"PassA": OrderedDict({"index": 0, "params": OrderedDict({"PA0_SP1": "A01b", "PA0_SP2": "A02a"})})}
                ),
                ["whatever"],
            ),
            (
                2,
                OrderedDict(
                    {"PassA": OrderedDict({"index": 0, "params": OrderedDict({"PA0_SP1": "A01a", "PA0_SP2": "A02b"})})}
                ),
                ["whatever"],
            ),
            (
                3,
                OrderedDict(
                    {"PassA": OrderedDict({"index": 0, "params": OrderedDict({"PA0_SP1": "A01b", "PA0_SP2": "A02b"})})}
                ),
                ["whatever"],
            ),
            (
                4,
                OrderedDict({"PassA": OrderedDict({"index": 1, "params": OrderedDict({"PA1_SP1": "A11a"})})}),
                ["whatever"],
            ),
            (
                5,
                OrderedDict({"PassA": OrderedDict({"index": 1, "params": OrderedDict({"PA1_SP1": "A11b"})})}),
                ["whatever"],
            ),
            (6, OrderedDict({"PassA": OrderedDict({"index": 2, "params": OrderedDict()})}), ["whatever"]),
            (7, OrderedDict({"PassA": OrderedDict({"index": 3, "params": OrderedDict()})}), ["whatever"]),
            (
                0,
                OrderedDict({"PassB": OrderedDict({"index": 0, "params": OrderedDict({"PB0_SP1": "B01a"})})}),
                ["model_id_1"],
            ),
            (
                1,
                OrderedDict({"PassB": OrderedDict({"index": 0, "params": OrderedDict({"PB0_SP1": "B01b"})})}),
                ["model_id_1"],
            ),
            (2, OrderedDict({"PassB": OrderedDict({"index": 1, "params": OrderedDict()})}), ["model_id_1"]),
            (0, OrderedDict({"PassC": OrderedDict({"index": 0, "params": OrderedDict()})}), ["model_id_5"]),
            (1, OrderedDict({"PassC": OrderedDict({"index": 1, "params": OrderedDict()})}), ["model_id_5"]),
        ]

        assert actual == expected
