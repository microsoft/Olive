# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Callable

from olive.evaluator.metric import AccuracySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.resource_path import StringName


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
            assert metric.name in ["accuracy", "hf_accuracy", "latency", "test"]

    def test_metrics_dump(self):
        user_script = "user_script.py"
        data_dir = "data_dir"
        post_process = "post_process"

        def dataloader_func():
            pass

        accuracy_metric_config = {
            "user_script": user_script,
            "post_processing_func": post_process,
            "data_dir": data_dir,
            "dataloader_func": dataloader_func,
        }
        accuracy_metric = Metric(
            name="accuracy",
            type=MetricType.ACCURACY,
            sub_types=[{"name": AccuracySubType.ACCURACY_SCORE}],
            user_config=accuracy_metric_config,
            data_config={
                "name": "wikitext2_test",
                "type": "HuggingfaceContainer",
                "params_config": {
                    "model_name": "openlm-research/open_llama_3b",
                    "task": "text-generation",
                    "data_name": "wikitext",
                    "subset": "wikitext-2-raw-v1",
                    "split": "test",
                    "input_cols": ["text"],
                    "seqlen": 2048,
                },
            },
        )
        accuracy_metric_json = accuracy_metric.model_dump()
        assert accuracy_metric_json["user_config"]["user_script"] == accuracy_metric_config["user_script"]
        assert (
            accuracy_metric_json["user_config"]["post_processing_func"]
            == accuracy_metric_config["post_processing_func"]
        )
        assert isinstance(accuracy_metric_json["user_config"]["data_dir"], StringName)
        assert str(accuracy_metric_json["user_config"]["data_dir"]) == accuracy_metric_config["data_dir"]
        assert isinstance(accuracy_metric_json["user_config"]["dataloader_func"], Callable)
        assert accuracy_metric_json["user_config"]["dataloader_func"] == accuracy_metric_config["dataloader_func"]
        assert accuracy_metric_json["data_config"]["name"] == "wikitext2_test"
