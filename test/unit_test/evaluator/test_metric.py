# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

class TestEvaluation:

    def test_metrics_config(self):
        # TODO move to run config tests
        metrics_config = [
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {
                        "name": "accuracy_score",
                        "priority_rank": 1,
                        "goal": {"type": "max-degradation", "value": 0.01}
                    },
                    {
                        "name": "auc",
                        "priority_rank": -1,
                        "metric_config": {
                            "reorder": True
                        },
                        "goal": {"type": "max-degradation", "value": 0.01}
                    }
                ]
            },
            {
                "name": "latency",
                "type": "latency",
                "sub_types": [
                    {
                        "name": "avg",
                        "priority_rank": 2,
                        "goal": {"type": "percent-min-improvement", "value": 20}
                    },
                    {"name": "max"},
                    {"name": "min"},
                ]
            },
            {
                "name": "test",
                "type": "custom",
                "sub_types": [
                    {
                        "name": "test",
                        "priority_rank": 2,
                        "higher_is_better": True,
                        "goal": {"type": "max-degradation", "value": 0.01}
                    },
                    {"name": "test"},
                    {"name": "test"},
                ]
            }
        ]

        from olive.evaluator.metric import MetricList
        metrics = MetricList.parse_obj(metrics_config)
        for metric in metrics:
            assert metric.name in ["accuracy", "latency", "test"]
