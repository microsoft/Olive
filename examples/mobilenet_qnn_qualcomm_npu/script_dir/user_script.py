import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset

from olive.common.utils import run_subprocess
from olive.evaluator.accuracy import AccuracyScore


class MobileNetDataset(Dataset):
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        data_file = data_dir / "data.npy"
        self.data = np.load(data_file)
        self.labels = None
        labels_file = data_dir / "labels.npy"
        if labels_file.exists():
            self.labels = np.load(labels_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1
        return {"input": data}, label


class MobileNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.dataset = MobileNetDataset(data_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.iter = iter(self.dataloader)

    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            batch = next(self.iter)[0]
        except StopIteration:
            return None

        batch = {k: v.detach().cpu().numpy() for k, v in batch.items()}
        return batch

    def rewind(self):
        self.iter = None


def evaluation_dataloader(data_dir, batch_size=1):
    dataset = MobileNetDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size)


def post_process(output):
    return output.argmax(axis=1)


def mobilenet_calibration_reader(data_dir, batch_size=1):
    return MobileNetCalibrationDataReader(data_dir, batch_size=batch_size)


def run_inference(model_path, input_data, device, type):
    from inference import get_path as get_inference_path

    inference_path = get_inference_path()
    try:
        qnn_env_path = os.environ["QNN_ENV_PATH"]
    except KeyError:
        raise ValueError("QNN_ENV_PATH environment variable is not set")
    try:
        qnn_lib_path = os.environ["QNN_LIB_PATH"]
    except KeyError:
        raise ValueError("QNN_LIB_PATH environment variable is not set")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        # save input data to file
        input_path = tmp_dir_path / "input.npz"
        input_data = {k: v.detach().cpu().numpy() for k, v in input_data.items()}
        np.savez(input_path, **input_data)

        # output path
        output_path = tmp_dir_path / "output.npy"

        # run inference
        command = (
            f"python {inference_path} --model_path {model_path} --input_path {input_path} --output_path"
            f" {output_path} --device {device} --type {type}"
        )
        env = deepcopy(os.environ)
        env["PATH"] = f"{qnn_env_path}{os.pathsep}{qnn_lib_path}"
        run_subprocess(command, env=env, check=True)

        # load output data
        output_data = np.load(output_path)
        return output_data


def eval_accuracy(model, data_dir, batch_size, device):
    dataloader = evaluation_dataloader(data_dir, batch_size=batch_size)

    preds = []
    targets = []
    for input_data, labels in dataloader:
        outputs = run_inference(model.model_path, input_data, device, "accuracy")
        outputs = torch.Tensor(outputs)
        outputs = post_process(outputs)

        preds.extend(outputs.tolist())
        targets.extend(labels.tolist())
    return AccuracyScore().evaluate(preds, targets)


def eval_latency(model, data_dir, batch_size, device):
    dataloader = evaluation_dataloader(data_dir, batch_size=batch_size)

    input_data, _ = next(iter(dataloader))
    latencies = run_inference(model.model_path, input_data, device, "latency")
    return round(sum(latencies) / len(latencies) * 1000, 5)
