# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.evaluator.metric_result import MetricResult
from olive.search.search_results import SearchResults

# ruff: noqa: PD011
# pylint: disable=W0212


class TestSearchResults:
    def test_empty(self):
        objectives = {
            "accuracy-accuracy_custom": {"goal": 0.75, "higher_is_better": True, "priority": 1},
            "latency-avg": {"goal": 24, "higher_is_better": False, "priority": 2},
            "latency-max": {"goal": 30, "higher_is_better": False, "priority": 3},
        }
        results = SearchResults(objectives)
        results.sort()

        assert results._sorted_indices == []
        assert results.get_next_best_result(-1) == (None, None, None)

    def test_sort(self):
        objectives = {
            "accuracy-accuracy_custom": {"goal": 0.75, "higher_is_better": True, "priority": 1},
            "latency-avg": {"goal": 24, "higher_is_better": False, "priority": 2},
            "latency-max": {"goal": 30, "higher_is_better": False, "priority": 3},
        }

        signal1 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.75,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 4.5, "priority": 2, "higher_is_better": False},
            }
        )
        signal2 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.78,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 55.9, "priority": 2, "higher_is_better": False},
            }
        )
        signal3 = MetricResult.parse_obj(
            {
                "accuracy-accuracy_custom": {
                    "value": 0.76,
                    "priority": 1,
                    "higher_is_better": True,
                },
                "latency-avg": {"value": 53.0, "priority": 2, "higher_is_better": False},
                "latency-max": {"value": 60.1, "priority": 3, "higher_is_better": False},
            }
        )

        signals = [
            (
                signal1,
                ["model_id_1"],
            ),
            None,
            None,
            (
                signal2,
                ["model_id_2"],
            ),
            None,
            None,
            None,
            (
                signal3,
                ["model_id_3"],
            ),
        ]

        results = SearchResults(objectives)
        for i, signal in enumerate(signals):
            if signal:
                results.record_feedback_signal(i, *signal)

        results.sort()

        assert results._sorted_indices == [3, 7, 0]

        next_best_spi = -1
        actual_order = []
        while next_best_spi is not None:
            next_best_spi, spi, model_ids = results.get_next_best_result(next_best_spi)
            if next_best_spi is not None:
                actual_order.append((next_best_spi, spi, model_ids))

        assert actual_order == [
            (0, 3, ["model_id_2"]),
            (1, 7, ["model_id_3"]),
            (2, 0, ["model_id_1"]),
        ]
