# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from test.integ_test.utils import download_azure_blob
from zipfile import ZipFile

from olive.common.config_utils import validate_config
from olive.data.config import DataComponentConfig, DataConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType

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


def _get_metric_data_config(name, dataset, post_process=None):
    data_config = DataConfig(
        name=name,
        user_script=str(_user_script),
        load_dataset_config=DataComponentConfig(
            type=dataset,
            params={"data_dir": str(_data_dir)},
        ),
    )
    if post_process:
        data_config.post_process_data_config = DataComponentConfig(type=post_process)
    return validate_config(data_config, DataConfig)


def get_accuracy_metric(post_process, dataset="mnist_dataset_for_local_eval"):
    sub_types = [{"name": AccuracySubType.ACCURACY_SCORE, "metric_config": {"task": "multiclass", "num_classes": 10}}]
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        data_config=_get_metric_data_config("accuracy_metric_data_config", dataset, post_process),
    )


def get_latency_metric(dataset="mnist_dataset_for_local_eval"):
    sub_types = [{"name": LatencySubType.AVG}]
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=sub_types,
        data_config=_get_metric_data_config("latency_metric_data_config", dataset),
    )


def get_hf_accuracy_metric(
    post_process="tiny_bert_post_process_for_local_eval", dataset="tiny_bert_dataset_for_local_eval"
):
    return get_accuracy_metric(post_process, dataset)


def get_hf_latency_metric(dataset="tiny_bert_dataset_for_local_eval"):
    return get_latency_metric(dataset)


def get_pytorch_model():
    download_path = str(_models_dir / "model.pt")
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": download_path,
    }
    download_azure_blob(**pytorch_model_config)
    return {"model_path": download_path}


def get_huggingface_model():
    return {
        "model_path": "hf-internal-testing/tiny-random-BertForSequenceClassification",
        "task": "text-classification",
    }


def get_onnx_model():
    download_path = str(_models_dir / "model.onnx")
    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": download_path,
    }
    download_azure_blob(**onnx_model_config)
    return {"model_path": download_path}


def get_openvino_model():
    download_path = str(_models_dir / "openvino.zip")
    openvino_model_config = {
        "container": "olivetest",
        "blob": "models/openvino.zip",
        "download_path": download_path,
    }
    download_azure_blob(**openvino_model_config)
    with ZipFile(download_path) as zip_ref:
        zip_ref.extractall(_models_dir)
    return {"model_path": str(_models_dir / "openvino")}


def delete_directories():
    shutil.rmtree(_data_dir)
    shutil.rmtree(_models_dir)
