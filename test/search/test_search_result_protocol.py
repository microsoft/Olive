# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.search.search_result_protocol import EvaluationSignal


class TestEvaluationSignalProtocol:
    """Tests for the EvaluationSignal protocol."""

    def test_dict_satisfies_protocol(self):
        """A plain dict supports __getitem__."""
        d = {"accuracy": 0.95}
        assert isinstance(d, EvaluationSignal)

    def test_custom_class_satisfies_protocol(self):
        """A custom class with __getitem__ satisfies the protocol."""

        class FakeResult:
            def __init__(self):
                self.data = {"key": "value"}

            def __getitem__(self, key):
                return self.data[key]

        assert isinstance(FakeResult(), EvaluationSignal)

    def test_class_without_getitem_fails(self):
        """A class missing __getitem__ should NOT satisfy the protocol."""

        class NoGetItem:
            pass

        assert not isinstance(NoGetItem(), EvaluationSignal)

    def test_metric_result_satisfies_protocol(self):
        """MetricResult (the concrete class) satisfies EvaluationSignal."""
        from olive.evaluator.metric_result import MetricResult, joint_metric_key

        mr = MetricResult.parse_obj(
            {
                joint_metric_key("accuracy", "accuracy_score"): {
                    "value": 0.95,
                    "priority": 1,
                    "higher_is_better": True,
                }
            }
        )
        assert isinstance(mr, EvaluationSignal)

        # Verify the operations work
        key = joint_metric_key("accuracy", "accuracy_score")
        assert key in mr
        assert mr[key].value == 0.95

    def test_protocol_used_in_search_results(self):
        """SearchResults can use any EvaluationSignal-compatible object."""
        from olive.search.search_results import SearchResults

        objectives = {
            "accuracy-accuracy_score": {
                "higher_is_better": True,
                "goal": 0.9,
                "priority": 1,
            }
        }
        sr = SearchResults(objectives)

        # Use a dict-like object instead of MetricResult
        class SimpleSignal:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __contains__(self, key):
                return key in self._data

        signal = SimpleSignal({"accuracy-accuracy_score": type("SubResult", (), {"value": 0.95})()})

        sr.record_feedback_signal(0, signal, ["model_1"])

        # Verify it works for goal checking
        assert sr.meets_goals(0) is True

    def test_protocol_is_runtime_checkable(self):
        """The protocol should be decorated with @runtime_checkable."""
        assert hasattr(EvaluationSignal, "__protocol_attrs__") or hasattr(EvaluationSignal, "__abstractmethods__")
        # Most reliable check: isinstance works without error
        assert isinstance({}, EvaluationSignal)
