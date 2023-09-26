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
                        "name": "auroc",
                        "priority": -1,
                        "metric_config": {"num_classes": 2},
                        "goal": {"type": "max-degradation", "value": 0.01},
                    },
                ],
            },
            {
                "name": "hf_accuracy",
                "type": "accuracy",
                "backend": "huggingface_metrics",
                "sub_types": [
                    {"name": "precision", "priority": -1, "goal": {"type": "max-degradation", "value": 0.01}},
                    {
                        "name": "recall",
                        "priority": -1,
                        "metric_config": {
                            "load_params": {"process_id": 0},
                            "compute_params": {"suffix": True},
                            "result_key": "recall",
                        },
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
            assert metric.user_config, "user_config should not be None anytime"
            assert metric.name in ["accuracy", "hf_accuracy", "latency", "test"]
