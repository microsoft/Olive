# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict

from olive.search.samplers.tpe_sampler import TPESampler
from olive.search.search_parameter import Categorical, Conditional
from olive.search.search_space import SearchSpace

# ruff: noqa: PD011


class TestTPESampler:
    def test_iteration(self):
        search_space = SearchSpace(
            [
                ("PassA", Categorical([1, 2])),
                ("PassB", Categorical([1, 2, 3])),
                ("PassC", Categorical(["a", "b"])),
                ("PassD", Categorical(["a", "b", "c"])),
            ]
        )
        objectives = {
            "accuracy-accuracy_custom": {"goal": 0.75, "higher_is_better": True, "priority": 1},
            "latency-avg": {"goal": 24, "higher_is_better": False, "priority": 2},
            "latency-max": {"goal": 30, "higher_is_better": False, "priority": 3},
        }

        config = {"seed": 101, "max_samples": 50}
        sampler = TPESampler(search_space, config, objectives)

        count = 0
        actual = []
        while not sampler.should_stop:
            actual.append(sampler.suggest())
            count += 1

        actual = [(search_point.index, search_point.values) for search_point in actual]
        expected = [
            (5, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (0, "a")})),
            (16, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (1, "b")})),
            (34, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (2, "c")})),
            (21, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (1, "b")})),
            (25, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (2, "c")})),
            (22, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (1, "b")})),
            (13, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (1, "b")})),
            (20, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (1, "b")})),
            (32, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (2, "c")})),
            (9, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (0, "a")})),
            (35, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (2, "c")})),
            (10, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (0, "a")})),
            (4, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (0, "a")})),
            (28, OrderedDict({"PassA": (0, 1), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (2, "c")})),
            (30, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (2, "c")})),
            (27, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (2, "c")})),
            (12, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (1, "b")})),
            (33, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (2, "c")})),
            (24, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (2, "c")})),
            (1, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (0, "a")})),
            (3, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (0, "a")})),
            (31, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (2, "c")})),
            (6, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (0, "a")})),
            (11, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (0, "a")})),
            (15, OrderedDict({"PassA": (1, 2), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (1, "b")})),
            (18, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (1, "b")})),
            (17, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (1, "b")})),
            (2, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (0, "a")})),
            (0, OrderedDict({"PassA": (0, 1), "PassB": (0, 1), "PassC": (0, "a"), "PassD": (0, "a")})),
            (14, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (1, "b")})),
            (7, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (0, "a")})),
            (29, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (0, "a"), "PassD": (2, "c")})),
            (8, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (1, "b"), "PassD": (0, "a")})),
            (26, OrderedDict({"PassA": (0, 1), "PassB": (1, 2), "PassC": (0, "a"), "PassD": (2, "c")})),
            (23, OrderedDict({"PassA": (1, 2), "PassB": (2, 3), "PassC": (1, "b"), "PassD": (1, "b")})),
            (19, OrderedDict({"PassA": (1, 2), "PassB": (0, 1), "PassC": (1, "b"), "PassD": (1, "b")})),
        ]

        assert count == 36
        assert actual == expected
        assert len({spi for spi, _ in expected}) == len(expected)

    def test_suggest(self):
        search_space = SearchSpace(
            [
                ("conversion", Categorical([SearchSpace([])])),
                ("transformers_optimization", Categorical([SearchSpace([])])),
                (
                    "quantization",
                    Categorical(
                        [
                            SearchSpace(
                                [
                                    ("quant_mode", Categorical(["dynamic", "static"])),
                                    ("weight_type", Categorical(["QInt8", "QUInt8"])),
                                    (
                                        "quant_format",
                                        Conditional(
                                            parents=("quant_mode",),
                                            support={("static",): Categorical(["QOperator", "QDQ"])},
                                            default=Conditional.get_ignored_choice(),
                                        ),
                                    ),
                                    (
                                        "activation_type",
                                        Conditional(
                                            parents=("quant_mode", "quant_format", "weight_type"),
                                            support={
                                                ("static", "QDQ", "QInt8"): Categorical(["QInt8"]),
                                                ("static", "QDQ", "QUInt8"): Categorical(["QUInt8"]),
                                                ("static", "QOperator", "QUInt8"): Categorical(["QUInt8"]),
                                                ("static", "QOperator", "QInt8"): Conditional.get_invalid_choice(),
                                            },
                                            default=Conditional.get_ignored_choice(),
                                        ),
                                    ),
                                    (
                                        "prepare_qnn_config",
                                        Conditional(
                                            parents=("quant_mode",),
                                            support={
                                                ("static",): Categorical([False]),
                                                ("dynamic",): Conditional.get_ignored_choice(),
                                            },
                                            default=Conditional.get_invalid_choice(),
                                        ),
                                    ),
                                    (
                                        "qnn_extra_options",
                                        Conditional(
                                            parents=("quant_mode",),
                                            support={
                                                ("static",): Categorical([None]),
                                                ("dynamic",): Conditional.get_ignored_choice(),
                                            },
                                            default=Conditional.get_invalid_choice(),
                                        ),
                                    ),
                                    (
                                        "MatMulConstBOnly",
                                        Conditional(
                                            parents=("quant_mode",),
                                            support={
                                                ("dynamic",): Categorical([True]),
                                                ("static",): Categorical([False]),
                                            },
                                            default=Conditional.get_invalid_choice(),
                                        ),
                                    ),
                                ]
                            )
                        ]
                    ),
                ),
                (
                    "session_params_tuning",
                    Categorical(
                        [
                            SearchSpace(
                                [("providers_list", Categorical(["OpenVINOExecutionProvider", "CPUExecutionProvider"]))]
                            )
                        ]
                    ),
                ),
            ]
        )
        objectives = {
            "accuracy-accuracy_custom": {"goal": 0.75, "higher_is_better": True, "priority": 1},
            "latency-avg": {"goal": 24, "higher_is_better": False, "priority": 2},
            "latency-max": {"goal": 30, "higher_is_better": False, "priority": 3},
        }

        config = {"seed": 101, "max_samples": 500}
        sampler = TPESampler(search_space, config, objectives)

        while not sampler.should_stop:
            sp = sampler.suggest()
            assert sp == search_space[sp.index]
