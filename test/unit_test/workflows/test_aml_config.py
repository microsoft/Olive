from copy import deepcopy
from unittest.mock import patch

import pytest

from olive.workflows.run.config import RunConfig


@pytest.fixture(autouse=True)
def config():
    config_json = {
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
                        "backend": "huggingface_metrics",
                        "sub_types": [
                            {"name": "accuracy", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                            {"name": "f1"},
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
                "config": {"model_type": "bert", "num_heads": 12, "hidden_size": 768, "float16": False},
            },
            "quantization": {"type": "OnnxQuantization"},
            "perf_tuning": {"type": "OrtPerfTuning"},
        },
        "engine": {
            "search_strategy": {
                "execution_order": "joint",
                "search_algorithm": "tpe",
                "search_algorithm_config": {"num_samples": 3, "seed": 0},
            },
            "evaluator": "common_evaluator",
            "execution_providers": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
            "cache_dir": "cache",
            "output_dir": "models/bert_ptq_cpu",
        },
    }
    return config_json


def test_aml_config(config):
    config_json = deepcopy(config)
    update_azureml_config(config_json)
    run_config = RunConfig.model_validate(config_json)
    assert run_config is not None
    if run_config.azureml_client:
        run_config.engine.azureml_client_config = run_config.azureml_client

    engine = run_config.engine.create_engine()
    assert engine is not None


@patch("olive.systems.azureml.aml_system.Environment")
def test_aml_system_engine_create(mock_env, config):
    config_json = deepcopy(config)

    config_patched = patch_config(config_json, "tpe", "joint", "aml_system")
    run_config = RunConfig.model_validate(config_patched)
    engine_cfg = run_config.engine.model_dump()
    assert engine_cfg["host"]["config"]["azureml_client_config"]
    engine = run_config.engine.create_engine()
    assert engine is not None


def patch_config(olive_config: dict, search_algorithm: str, execution_order: str, system: str, is_gpu: bool = False):
    # set default logger severity
    olive_config["engine"]["log_severity_level"] = 0
    # set clean cache
    olive_config["engine"]["clean_cache"] = True

    # update search strategy
    olive_config["engine"]["search_strategy"]["search_algorithm"] = search_algorithm
    if search_algorithm == "random" or search_algorithm == "tpe":
        olive_config["engine"]["search_strategy"]["search_algorithm_config"] = {"num_samples": 3, "seed": 0}
    olive_config["engine"]["search_strategy"]["execution_order"] = execution_order

    update_azureml_config(olive_config)
    if system == "aml_system":
        # set aml_system
        set_aml_system(olive_config, is_gpu=is_gpu)
        olive_config["engine"]["host"] = system
        olive_config["engine"]["target"] = system
    elif system == "docker_system":
        # set docker_system
        set_docker_system(olive_config)
        olive_config["engine"]["target"] = system
        # reduce agent size for docker system

        # as our docker image is big, we need to reduce the agent size to avoid timeout
        # for the docker system test, we skip to search for transformers optimization as
        # it is tested in other olive system tests
        olive_config["passes"]["transformers_optimization"]["disable_search"] = True
        olive_config["engine"]["search_strategy"]["search_algorithm_config"]["num_samples"] = 2

    return olive_config


def update_azureml_config(olive_config):
    """Update the azureml config in the olive config."""
    olive_config["azureml_client"] = {
        "subscription_id": "3905431d-c062-4c17-8fd9-c51f89f334c4",
        "resource_group": "olive",
        "workspace_name": "olive-aml-workspace",
    }


def set_aml_system(olive_config, is_gpu=False):
    """Set the aml_system in the olive config."""
    if "systems" not in olive_config:
        olive_config["systems"] = {}

    if is_gpu:
        olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "config": {
                "accelerators": ["GPU"],
                "aml_compute": "gpu-cluster",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04:20230608.v1",
                    "conda_file_path": "conda_gpu.yaml",
                },
                "is_dev": True,
            },
        }

    else:
        olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "config": {
                "accelerators": ["CPU"],
                "aml_compute": "cpu-cluster",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                    "conda_file_path": "conda.yaml",
                },
                "is_dev": True,
            },
        }


def set_docker_system(olive_config):
    """Set the docker_system in the olive config."""
    if "systems" not in olive_config:
        olive_config["systems"] = {}

    olive_config["systems"]["docker_system"] = {
        "type": "Docker",
        "config": {
            "local_docker_config": {
                "image_name": "olive-image",
                "build_context_path": "docker",
                "dockerfile": "Dockerfile",
            },
            "is_dev": True,
        },
    }
