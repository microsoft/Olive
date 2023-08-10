from olive.workflows.run.config import RunConfig


def test_aml_config():
    config = {
        "azureml_client": {
            "subscription_id": "3905431d-c062-4c17-8fd9-c51f89f334c4",
            "resource_group": "olive",
            "workspace_name": "olive-aml-workspace",
        },
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "model_path": {"type": "azureml_model", "config": {"name": "bert-hf", "version": "3"}},
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
                },
            },
        },
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {"name": "accuracy", "type": "accuracy", "sub_types": [{"name": "accuracy_score", "priority": 1}]},
                    {"name": "latency", "type": "latency", "sub_types": [{"name": "avg", "priority": 2}]},
                ]
            }
        },
        "passes": {
            "conversion": {"type": "OnnxConversion", "config": {"target_opset": 13}},
            "transformers_optimization": {
                "type": "OrtTransformersOptimization",
                "config": {
                    "model_type": "bert",
                    "num_heads": 12,
                    "hidden_size": 768,
                    "float16": False,
                },
            },
            "quantization": {"type": "OnnxQuantization"},
            "perf_tuning": {"type": "OrtPerfTuning"},
        },
        "engine": {
            "search_strategy": False,
            "evaluator": "common_evaluator",
            "cache_dir": "cache",
            "output_dir": "models/bert_hf_cpu_aml",
        },
    }
    run_config = RunConfig.model_validate(config)
    assert run_config is not None
    if run_config.azureml_client:
        run_config.engine.azureml_client_config = run_config.azureml_client

    engine = run_config.engine.create_engine()
    assert engine is not None
