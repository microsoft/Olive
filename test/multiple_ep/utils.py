# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.integ_test.utils import download_azure_blob

from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.data.config import DataComponentConfig, DataConfig
from olive.engine import Engine
from olive.evaluator.metric import LatencySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.passes.onnx.session_params_tuning import OrtSessionParamsTuning
from olive.systems.accelerator_creator import create_accelerators

# pylint: disable=redefined-outer-name


def get_directories():
    current_dir = Path(__file__).resolve().parent

    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    user_script = current_dir / "user_script.py"

    return current_dir, models_dir, data_dir, user_script


_current_dir, _models_dir, _data_dir, _user_script = get_directories()


def get_latency_metric():
    data_config = DataConfig(
        name="latency_metric_data_config",
        user_script=str(_user_script),
        load_dataset_config=DataComponentConfig(
            type="mnist_dataset_for_multiple_ep",
            params={"data_dir": str(_data_dir)},
        ),
    )
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
        data_config=data_config,
    )


def download_models():
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": str(_models_dir / "model.pt"),
    }
    download_azure_blob(**pytorch_model_config)

    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": str(_models_dir / "model.onnx"),
    }
    download_azure_blob(**onnx_model_config)


def download_data():
    datasets.MNIST(str(_data_dir), download=True, transform=ToTensor())


def get_onnx_model():
    return str(_models_dir / "model.onnx")


def create_and_run_workflow(tmp_path, system_config, model_config, metric, only_target=False):
    # use the olive managed python environment as the test environment
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    evaluator_config = OliveEvaluatorConfig(metrics=[metric])

    cache = tmp_path / "cache"
    cache.mkdir()
    config = {
        "cache_config": {
            "cache_dir": cache,
        },
        "target": system_config,
        "host": system_config if not only_target else None,
        "evaluator": evaluator_config,
    }
    engine = Engine(**config)
    engine.register(OrtSessionParamsTuning)
    accelerator_specs = create_accelerators(system_config)
    output = engine.run(
        model_config,
        accelerator_specs,
        output_dir=output_dir,
        evaluate_input_model=True,
    )

    results = [next(iter(output[accelerator].nodes.values())) for accelerator in accelerator_specs]
    return tuple(results)
