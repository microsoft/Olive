# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.evaluator.olive_evaluator import OliveEvaluatorConfig


class TestEvaluation:
    def test_metrics_config(self):
        metrics_config = [
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                    {
                        "name": "auc",
                        "priority": -1,
                        "metric_config": {"reorder": True},
                        "goal": {"type": "max-degradation", "value": 0.01},
                    },
                ],
            },
            {
                "name": "latency",
                "type": "latency",
                "sub_types": [
                    {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 20}},
                    {"name": "max"},
                    {"name": "min"},
                ],
            },
            {
                "name": "test",
                "type": "custom",
                "sub_types": [
                    {
                        "name": "test",
                        "priority": 3,
                        "higher_is_better": True,
                        "goal": {"type": "max-degradation", "value": 0.01},
                    },
                    {"name": "test"},
                    {"name": "test"},
                ],
            },
        ]

        metrics = OliveEvaluatorConfig(metrics=metrics_config).metrics
        for metric in metrics:
            assert metric.name in ["accuracy", "latency", "test"]
