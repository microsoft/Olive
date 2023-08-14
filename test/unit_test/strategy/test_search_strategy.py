from olive.workflows.run.config import RunConfig


def test_search_strategy_serialization():
    config = {
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "hf_config": {
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                    "dataset": {
                        "data_name": "glue",
                        "subset": "mrpc",
                        "split": "validation",
                        "input_cols": ["sentence1", "sentence2"],
                        "label_cols": ["label"],
                        "batch_size": 1,
                    },
                }
            },
        },
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {
                        "name": "accuracy",
                        "type": "accuracy",
                        "sub_types": [
                            {
                                "name": "accuracy_score",
                                "priority": 1,
                                "goal": {"type": "max-degradation", "value": 0.01},
                            },
                            {"name": "f1_score", "metric_config": {"multiclass": False}},
                            {"name": "auroc", "metric_config": {"num_classes": 2}},
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
                ]
            }
        },
        "passes": {
            "conversion": {"type": "OnnxConversion", "config": {"target_opset": 13}},
            "transformers_optimization": {
                "type": "OrtTransformersOptimization",
                "config": {"model_type": "bert", "num_heads": 12, "hidden_size": 768, "float16": True},
            },
            "perf_tuning": {"type": "OrtPerfTuning", "config": {"enable_cuda_graph": True}},
        },
        "engine": {
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "tpe",
                "search_algorithm_config": {"num_samples": 3, "seed": 0},
            },
            "evaluator": "common_evaluator",
            "execution_providers": ["CUDAExecutionProvider"],
        },
    }
    config = RunConfig.model_validate(config)
    assert config is not None

    search_strategy_json = config.engine.search_strategy.model_dump()
    assert search_strategy_json["search_algorithm_config"]["num_samples"] == 3
