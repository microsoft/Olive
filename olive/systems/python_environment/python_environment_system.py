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
from olive.evaluator.metric import (
    Metric,
    MetricResult,
    MetricType,
    flatten_metric_result,
    get_latency_config_from_metric,
)
from olive.evaluator.olive_evaluator import OliveEvaluator, OnnxEvaluator
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec, Device
from olive.model import OliveModel, ONNXModel
from olive.passes.olive_pass import Pass
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import PythonEnvironmentTargetUserConfig

logger = logging.getLogger(__name__)


class PythonEnvironmentSystem(OliveSystem):
    system_type = SystemType.PythonEnvironment

    def __init__(
        self,
        python_environment_path: Union[Path, str],
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
        accelerators: List[str] = None,
    ):
        super().__init__(accelerators=accelerators)
        self.config = PythonEnvironmentTargetUserConfig(
            python_environment_path=python_environment_path,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
            accelerators=accelerators,
        )
        self.environ = deepcopy(os.environ)
        if self.config.environment_variables:
            self.environ.update(self.config.environment_variables)
        if self.config.prepend_to_path:
            self.environ["PATH"] = os.pathsep.join(self.config.prepend_to_path) + os.pathsep + self.environ["PATH"]
        self.environ["PATH"] = str(self.config.python_environment_path) + os.pathsep + self.environ["PATH"]

        # available eps. This will be populated the first time self.get_supported_execution_providers() is called.
        # used for caching the available eps
        self.available_eps = None

        # path to inference script
        self.inference_path = Path(__file__).parent.resolve() / "inference_runner.py"
        self.device = self.accelerators[0] if self.accelerators else Device.CPU

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

    def evaluate_model(self, model: OliveModel, metrics: List[Metric], accelerator: AcceleratorSpec) -> MetricResult:
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
        return flatten_metric_result(metrics_res)

    def evaluate_accuracy(self, model: ONNXModel, metric: Metric) -> float:
        """
        Evaluate the accuracy of the model.
        """
        dataloader, post_func, _ = OliveEvaluator.get_user_config(metric)

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
            with open(inference_settings_path, "wb") as f:
                pickle.dump(inference_settings, f)

            num_batches = 0
            for idx, (input_data, labels) in enumerate(dataloader):
                # save input data to npz
                input_dict = OnnxEvaluator.format_input(input_data, io_config)
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

        return OliveEvaluator.compute_accuracy(metric, preds, targets)

    def evaluate_latency(self, model: ONNXModel, metric: Metric) -> float:
        """
        Evaluate the latency of the model.
        """
        dataloader, _, _ = OliveEvaluator.get_user_config(metric)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)

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
            with open(inference_settings_path, "wb") as f:
                pickle.dump(inference_settings, f)

            # save input data to npz
            input_data, _ = next(iter(dataloader))
            input_dict = OnnxEvaluator.format_input(input_data, io_config)
            np.savez(input_dir / "input.npz", **input_dict)

            # run inference
            command = (
                f"python {self.inference_path} --type {metric.type} --model_path"
                f" {model.model_path} --inference_settings_path {inference_settings_path} --input_dir"
                f" {input_dir} --output_dir  {output_dir} --warmup_num {warmup_num} --repeat_test_num"
                f" {repeat_test_num} --sleep_num {sleep_num}"
            )
            if metric.user_config.io_bind:
                command += f" --io_bind --device {self.device}"
            run_subprocess(command, env=self.environ, check=True)

            # load output
            latencies = np.load(output_dir / "output.npy")

        return OliveEvaluator.compute_latency(metric, latencies)

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
            inference_settings["execution_provider"] = self.get_default_execution_provider(model)

        return inference_settings

    def get_supported_execution_providers(self) -> List[str]:
        """
        Get the available execution providers.
        """
        if self.available_eps:
            return self.available_eps

        with tempfile.TemporaryDirectory() as temp_dir:
            available_eps_path = Path(__file__).parent.resolve() / "available_eps.py"
            output_path = Path(temp_dir).resolve() / "available_eps.pb"
            run_subprocess(
                f"python {available_eps_path} --output_path {output_path}",
                env=self.environ,
                check=True,
            )
            with output_path.open("rb") as f:
                available_eps = pickle.load(f)
            self.available_eps = available_eps
            return available_eps

    def get_execution_providers(self) -> List[str]:
        """
        Get the execution providers for the device.
        """
        available_eps = self.get_supported_execution_providers()
        return AcceleratorLookup.get_execution_providers_for_device_by_available_providers(self.device, available_eps)

    def get_default_execution_provider(self, model: ONNXModel) -> List[str]:
        """
        Get the default execution provider for the model.
        """
        # return first available ep as ort default ep
        available_providers = self.get_execution_providers()
        for ep in available_providers:
            if self.is_valid_ep(ep, model):
                return [ep]
        return ["CPUExecutionProvider"]

    def is_valid_ep(self, ep: str, model: ONNXModel) -> bool:
        """
        Check if the execution provider is valid for the model.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            is_valid_ep_path = Path(__file__).parent.resolve() / "is_valid_ep.py"
            output_path = Path(temp_dir).resolve() / "result.pb"
            run_subprocess(
                " ".join(
                    [
                        "python",
                        str(is_valid_ep_path),
                        "--model_path",
                        str(model.model_path),
                        "--ep",
                        ep,
                        "--output_path",
                        str(output_path),
                    ]
                ),
                env=self.environ,
                check=True,
            )
            with output_path.open("rb") as f:
                result = pickle.load(f)
            if result["valid"]:
                return True
            else:
                logger.warning(
                    f"Error: {result['error']} Olive will ignore this {ep}."
                    + f"Please make sure the environment with {ep} has the required dependencies."
                )
                return False
