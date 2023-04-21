# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import pickle
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from olive.common.utils import run_subprocess
from olive.evaluator.evaluation import compute_accuracy, compute_latency, format_onnx_input, get_user_config
from olive.evaluator.metric import Metric, MetricType
from olive.model import OliveModel, ONNXModel
from olive.passes.olive_pass import Pass
from olive.systems.common import Device, SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import PythonEnvironmentTargetUserConfig

logger = logging.getLogger(__name__)


class PythonEnvironmentSystem(OliveSystem):
    system_type = SystemType.PythonEnvironment

    def __init__(
        self,
        python_environment_path: Union[Path, str],
        device: Device = Device.CPU,
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
    ):
        super().__init__()
        self.config = PythonEnvironmentTargetUserConfig(
            python_environment_path=python_environment_path,
            device=device,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
        )
        self.environ = deepcopy(os.environ)
        if self.config.environment_variables:
            self.environ.update(self.config.environment_variables)
        if self.config.prepend_to_path:
            self.environ["PATH"] = os.pathsep.join(self.config.prepend_to_path) + os.pathsep + self.environ["PATH"]
        self.environ["PATH"] = str(self.config.python_environment_path) + os.pathsep + self.environ["PATH"]

        # get available EPs
        self.available_eps = self.get_available_eps()

        # path to inference script
        self.inference_path = Path(__file__).parent.resolve() / "inference_runner.py"

    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        raise ValueError("PythonEnvironmentSystem does not support running passes.")

    def evaluate_model(self, model: OliveModel, metrics: List[Metric]) -> Dict[str, Any]:
        """
        Evaluate the model
        """
        if not isinstance(model, ONNXModel):
            raise ValueError("PythonEnvironmentSystem can only evaluate ONNXModel.")

        # check if custom metric is present
        if any(metric.type == MetricType.CUSTOM for metric in metrics):
            raise ValueError("PythonEnvironmentSystem does not support custom metrics.")

        metrics_res = {}
        for metric in metrics:
            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = self.evaluate_accuracy(model, metric)
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = self.evaluate_latency(model, metric)
        return metrics_res

    def evaluate_accuracy(self, model: ONNXModel, metric: Metric) -> float:
        """
        Evaluate the accuracy of the model.
        """
        dataloader, post_func, _ = get_user_config(metric.user_config)

        preds = []
        targets = []
        inference_settings = self.get_inference_settings(model, metric)
        io_config = model.get_io_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            # create input and output dir
            input_dir = tmp_dir_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir = tmp_dir_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # save inference settings
            inference_settings_path = tmp_dir_path / "inference_settings.pb"
            pickle.dump(inference_settings, open(inference_settings_path, "wb"))

            num_batches = 0
            for idx, (input_data, labels) in enumerate(dataloader):
                # save input data to npz
                input_dict = format_onnx_input(input_data, io_config)
                input_path = input_dir / f"input_{idx}.npz"
                np.savez(input_path, **input_dict)
                # save labels
                targets.extend(labels.data.tolist())
                num_batches += 1

            # run inference
            command = (
                f"python {self.inference_path} --type {metric.type} --model_path"
                f" {model.model_path} --inference_settings_path {inference_settings_path} --input_dir"
                f" {input_dir} --num_batches {num_batches} --output_dir  {output_dir}"
            )
            run_subprocess(command, env=self.environ, check=True)

            # load output
            output_names = io_config["output_names"]
            for idx in range(num_batches):
                output_path = output_dir / f"output_{idx}.npy"
                output = np.load(output_path)
                output = torch.Tensor(output[0] if len(output_names) == 1 else output)
                if post_func:
                    output = post_func(output)
                preds.extend(output.tolist())

        return compute_accuracy(preds, targets, metric.sub_type, metric.metric_config)

    def evaluate_latency(self, model: ONNXModel, metric: Metric) -> float:
        """
        Evaluate the latency of the model.
        """
        dataloader, _, _ = get_user_config(metric.user_config)
        warmup_num = metric.metric_config.warmup_num
        repeat_test_num = metric.metric_config.repeat_test_num
        sleep_num = metric.metric_config.sleep_num

        latencies = []
        inference_settings = self.get_inference_settings(model, metric)
        io_config = model.get_io_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            # create input and output dir
            input_dir = tmp_dir_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir = tmp_dir_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # save inference settings
            inference_settings_path = tmp_dir_path / "inference_settings.pb"
            pickle.dump(inference_settings, open(inference_settings_path, "wb"))

            # save input data to npz
            input_data, _ = next(iter(dataloader))
            input_dict = format_onnx_input(input_data, io_config)
            np.savez(input_dir / "input.npz", **input_dict)

            # run inference
            command = (
                f"python {self.inference_path} --type {metric.type} --model_path"
                f" {model.model_path} --inference_settings_path {inference_settings_path} --input_dir"
                f" {input_dir} --output_dir  {output_dir} --warmup_num {warmup_num} --repeat_test_num"
                f" {repeat_test_num} --sleep_num {sleep_num} --io_bind {metric.user_config.io_bind} --device"
                f" {self.config.device}"
            )
            run_subprocess(command, env=self.environ, check=True)

            # load output
            latencies = np.load(output_dir / "output.npy")

        return compute_latency(latencies, metric.sub_type)

    def get_inference_settings(self, model: ONNXModel, metric: Metric) -> Dict[str, Any]:
        """
        Get the model inference settings.
        """
        metric_inference_settings = metric.user_config.inference_settings
        inference_settings = (
            metric_inference_settings.get(model.framework.lower()) if metric_inference_settings else None
        )
        inference_settings = inference_settings or model.inference_settings or {}
        inference_settings = deepcopy(inference_settings)

        # if user doesn't not provide ep list, use default value([ep]). Otherwise, use the user's ep list
        if not inference_settings.get("execution_provider"):
            inference_settings["execution_provider"] = self.get_default_ep()

        return inference_settings

    def get_available_eps(self) -> List[str]:
        """
        Get the available execution providers.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            available_eps_path = Path(__file__).parent.resolve() / "available_eps.py"
            output_path = Path(temp_dir).resolve() / "available_eps.pb"
            run_subprocess(
                f"python {available_eps_path} --output_path {output_path}",
                env=self.environ,
                check=True,
            )
            with open(output_path, "rb") as f:
                return pickle.load(f)

    def get_default_ep(self) -> List[str]:
        """
        Get the default execution provider.
        """
        eps_per_device = ONNXModel.EXECUTION_PROVIDERS.get(self.config.device)
        eps = []
        if eps_per_device:
            for ep in self.available_eps:
                if ep in eps_per_device:
                    eps.append(ep)
        eps = eps or self.available_eps
        return eps[0]
