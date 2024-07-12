# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from olive.common.config_utils import validate_config
from olive.constants import Framework
from olive.data.component.dataset import DummyDataset
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.evaluator.metric_config import MetricGoal
from olive.model import ModelConfig, ONNXModelHandler, PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict

ONNX_MODEL_PATH = Path(__file__).absolute().parent / "dummy_model.onnx"


class DummyModel(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.fc1 = nn.Linear(batch_size, 10)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))


# TODO(shaahji): Remove this once perf_tuning pass supports DataConfig
def create_dummy_dataloader(data_dir, batch_size=1, max_samples=32, **kwargs):
    return DataLoader(DummyDataset([(batch_size or 1, 1)], max_samples=max_samples), batch_size=None)


def pytorch_model_loader(model_path):
    return DummyModel().eval()


def get_pytorch_model_io_config(batch_size=1):
    return {"input_names": ["input"], "output_names": ["output"], "input_shapes": [(batch_size, 1)]}


def get_pytorch_model_config(batch_size=1):
    config = {
        "type": "PyTorchModel",
        "config": {
            "model_loader": pytorch_model_loader,
            "io_config": get_pytorch_model_io_config(batch_size),
        },
    }
    return ModelConfig.parse_obj(config)


def get_pytorch_model(batch_size=1):
    return PyTorchModelHandler(
        model_loader=pytorch_model_loader,
        model_path=None,
        io_config=get_pytorch_model_io_config(batch_size),
    )


def get_hf_model():
    return PyTorchModelHandler(
        hf_config={
            "model_name": "hf-internal-testing/tiny-random-gptj",
            "task": "text-generation",
        }
    )


def get_hf_model_with_past():
    return PyTorchModelHandler(
        hf_config={
            "model_name": "hf-internal-testing/tiny-random-gptj",
            "task": "text-generation",
            "feature": "causal-lm-with-past",
        }
    )


def get_pytorch_model_dummy_input(model=None, batch_size=1):
    return torch.randn(batch_size, 1)


def create_onnx_model_file():
    pytorch_model = pytorch_model_loader(model_path=None)
    dummy_input = get_pytorch_model_dummy_input(pytorch_model)
    io_config = get_pytorch_model_io_config()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        ONNX_MODEL_PATH,
        opset_version=10,
        input_names=io_config["input_names"],
        output_names=io_config["output_names"],
    )


def create_onnx_model_with_dynamic_axis(onnx_model_path, batch_size=1):
    pytorch_model = pytorch_model_loader(model_path=None)
    dummy_input = get_pytorch_model_dummy_input(pytorch_model, batch_size)
    io_config = get_pytorch_model_io_config()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_model_path,
        opset_version=10,
        input_names=io_config["input_names"],
        output_names=io_config["output_names"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def get_onnx_model_config():
    return ModelConfig.parse_obj({"type": "ONNXModel", "config": {"model_path": str(ONNX_MODEL_PATH)}})


def get_composite_onnx_model_config():
    onnx_model_config = get_onnx_model_config().dict()
    return ModelConfig.parse_obj(
        {
            "type": "CompositeModel",
            "config": {
                "model_components": [onnx_model_config, onnx_model_config],
                "model_component_names": "test_component_name",
            },
        }
    )


def get_onnx_model():
    return ONNXModelHandler(model_path=str(ONNX_MODEL_PATH))


def delete_onnx_model_files():
    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)


def get_mock_snpe_model():
    olive_model = MagicMock()
    olive_model.framework = Framework.SNPE
    return olive_model


def get_mock_openvino_model():
    olive_model = MagicMock()
    olive_model.framework = Framework.OPENVINO
    return olive_model


def _get_dummy_data_config(name, input_shapes, max_samples=1):
    data_config = DataConfig(
        name=name,
        type="DummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "input_shapes": input_shapes,
                "max_samples": max_samples,
            }
        ),
        post_process_data_config=DataComponentConfig(type="text_classification_post_process"),
    )
    return validate_config(data_config, DataConfig)


def get_accuracy_metric(
    *acc_subtype,
    user_config=None,
    backend="torch_metrics",
    goal_type="threshold",
    goal_value=0.99,
):
    accuracy_score_metric_config = {"task": "multiclass", "num_classes": 10}
    sub_types = [
        {
            "name": sub,
            "metric_config": accuracy_score_metric_config if sub == "accuracy_score" else {},
            "goal": MetricGoal(type=goal_type, value=goal_value),
        }
        for sub in acc_subtype
    ]
    sub_types[0]["priority"] = 1
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        user_config=user_config,
        backend=backend,
        data_config=_get_dummy_data_config("accuracy_metric_data_config", [[1, 1]]),
    )


def get_glue_accuracy_metric():
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=[{"name": AccuracySubType.ACCURACY_SCORE}],
        data_config=get_glue_huggingface_data_config(),
    )


def get_glue_latency_metric():
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
        data_config=get_glue_huggingface_data_config(),
    )


def get_custom_metric(user_config=None):
    user_script_path = str(Path(__file__).absolute().parent / "assets" / "user_script.py")
    return Metric(
        name="custom",
        type=MetricType.CUSTOM,
        sub_types=[{"name": "custom"}],
        user_config=user_config or {"evaluate_func": "eval_func", "user_script": user_script_path},
    )


def get_custom_metric_no_eval():
    custom_metric = get_custom_metric()
    custom_metric.user_config.evaluate_func = None
    return custom_metric


def get_latency_metric(*lat_subtype, user_config=None):
    sub_types = [{"name": sub} for sub in lat_subtype]
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=sub_types,
        user_config=user_config,
        data_config=_get_dummy_data_config("latency_metric_data_config", [[1, 1]]),
    )


def get_throughput_metric(*lat_subtype, user_config=None):
    sub_types = [{"name": sub} for sub in lat_subtype]
    return Metric(
        name="throughput",
        type=MetricType.THROUGHPUT,
        sub_types=sub_types,
        user_config=user_config,
        data_config=_get_dummy_data_config("throughput_metric_data_config", [[1, 1]]),
    )


def get_onnxconversion_pass(ignore_pass_config=True, target_opset=13):
    from olive.passes.onnx.conversion import OnnxConversion

    onnx_conversion_config = {"target_opset": target_opset}
    p = create_pass_from_dict(OnnxConversion, onnx_conversion_config)
    if ignore_pass_config:
        return p
    pass_config = p.config_at_search_point({})
    pass_config = p.serialize_config(pass_config)
    return p, pass_config


def get_onnx_dynamic_quantization_pass(disable_search=False):
    from olive.passes.onnx.quantization import OnnxDynamicQuantization

    return create_pass_from_dict(OnnxDynamicQuantization, disable_search=disable_search)


def get_data_config():
    @Registry.register_dataset("test_dataset")
    def _test_dataset(data_dir, test_value): ...

    @Registry.register_dataloader()
    def _test_dataloader(dataset, test_value): ...

    @Registry.register_pre_process()
    def _pre_process(dataset, test_value): ...

    @Registry.register_post_process()
    def _post_process(output, test_value): ...

    return DataConfig(
        name="test_data_config",
        load_dataset_config=DataComponentConfig(
            type="test_dataset",  # renamed by Registry.register_dataset
            params={"test_value": "test_value"},
        ),
        dataloader_config=DataComponentConfig(
            type="_test_dataloader",  # This is the key to get dataloader
            params={"test_value": "test_value"},
        ),
    )


def get_glue_huggingface_data_config():
    return DataConfig(
        name="glue_huggingface_data_config",
        type="HuggingfaceContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "data_name": "glue",
                "subset": "mrpc",
                "split": "validation",
                "batch_size": 1,
            }
        ),
        pre_process_data_config=DataComponentConfig(
            params={
                "model_name": "Intel/bert-base-uncased-mrpc",
                "task": "text-classification",
                "input_cols": ["sentence1", "sentence2"],
                "label_cols": ["label"],
            }
        ),
        post_process_data_config=DataComponentConfig(
            params={
                "model_name": "Intel/bert-base-uncased-mrpc",
                "task": "text-classification",
            }
        ),
    )


def get_transformer_dummy_input_data_config():
    return DataConfig(
        name="transformer_token_dummy_data",
        type="TransformersTokenDummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "model_name": "Intel/bert-base-uncased-mrpc",
                "use_step": True,
            }
        ),
        dataloader_config=DataComponentConfig(
            params={
                "batch_size": 2,
            }
        ),
    )


def create_raw_data(raw_data_dir, input_names, input_shapes, input_types=None, num_samples=1):
    data_dir = Path(raw_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    input_types = input_types or ["float32"] * len(input_names)

    num_samples_digits = len(str(num_samples))

    data = {}
    for input_name, input_shape, input_type in zip(input_names, input_shapes, input_types):
        data[input_name] = []
        input_dir = data_dir / input_name
        input_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples):
            data_i = np.random.rand(*input_shape).astype(input_type)
            data_i.tofile(input_dir / f"{i}.bin".zfill(num_samples_digits + 4))
            data[input_name].append(data_i)

    return data
