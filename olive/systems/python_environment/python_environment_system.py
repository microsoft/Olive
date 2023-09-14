# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import json
import logging
import os
import pickle
import platform
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
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveModelOutput, OnnxEvaluator
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec, Device
from olive.model import ModelConfig, ONNXModel
from olive.passes.olive_pass import Pass
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import PythonEnvironmentTargetUserConfig
from olive.systems.utils import get_package_name

logger = logging.getLogger(__name__)


class PythonEnvironmentSystem(OliveSystem):
    system_type = SystemType.PythonEnvironment

    def __init__(
        self,
        python_environment_path: Union[Path, str] = None,
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
        accelerators: List[str] = None,
        olive_managed_env: bool = False,
        requirements_file: Union[Path, str] = None,
    ):
        super().__init__(accelerators=accelerators, olive_managed_env=olive_managed_env)
        self.config = PythonEnvironmentTargetUserConfig(
            python_environment_path=python_environment_path,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
            accelerators=accelerators,
            olive_managed_env=olive_managed_env,
            requirements_file=requirements_file,
        )
        self.environ = deepcopy(os.environ)
        if self.config.environment_variables:
            self.environ.update(self.config.environment_variables)
        if self.config.prepend_to_path:
            self.environ["PATH"] = os.pathsep.join(self.config.prepend_to_path) + os.pathsep + self.environ["PATH"]
        if self.config.python_environment_path:
            self.environ["PATH"] = str(self.config.python_environment_path) + os.pathsep + self.environ["PATH"]
        if self.config.olive_managed_env:
            if platform.system() == "Linux":
                temp_dir = os.path.join(os.environ.get("HOME", ""), "tmp")
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                self.environ["TMPDIR"] = temp_dir
            else:
                self.environ["TMPDIR"] = tempfile.TemporaryDirectory().name

        # available eps. This will be populated the first time self.get_supported_execution_providers() is called.
        # used for caching the available eps
        self.available_eps = None

        # path to inference script
        self.inference_path = Path(__file__).parent.resolve() / "inference_runner.py"
        self.pass_path = Path(__file__).parent.resolve() / "pass_runner.py"
        self.device = self.accelerators[0] if self.accelerators else Device.CPU

    def run_pass(
        self,
        the_pass: Pass,
        model_config: ModelConfig,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """
        Run the pass on the model at a specific point in the search space.
        """
        model_config_json = model_config.to_json()
        pass_config = the_pass.to_json()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            model_json_path = tmp_dir_path / "model.json"
            pass_json_path = tmp_dir_path / "pass.json"
            output_model_json_path = tmp_dir_path / "output_model.json"

            with model_json_path.open("w") as f:
                json.dump(model_config_json, f, indent=4)
            with pass_json_path.open("w") as f:
                json.dump(pass_config, f, indent=4)

            # run pass
            command = (
                f"python {self.pass_path} --model_json_path {model_json_path} --pass_json_path {pass_json_path}"
                f" --output_model_path {output_model_path} --output_model_json_path {output_model_json_path}"
            )
            if point:
                point_json_path = tmp_dir_path / "point.json"
                with point_json_path.open("w") as f:
                    point = point or {}
                    json.dump(point, f, indent=4)
                command += f" --point_json_path {point_json_path}"
            if data_root:
                command += f" --data_root {data_root}"

            run_subprocess(command, env=self.environ, check=True)

            with open(output_model_json_path, "r") as f:
                model_json = json.load(f)
                output_model = ModelConfig.from_json(model_json)

        return output_model

    def evaluate_model(
        self, model_config: ModelConfig, data_root: str, metrics: List[Metric], accelerator: AcceleratorSpec
    ) -> MetricResult:
        """
        Evaluate the model
        """
        if not model_config.type.lower() == "ONNXModel".lower():
            raise ValueError("PythonEnvironmentSystem can only evaluate ONNXModel.")

        # check if custom metric is present
        if any(metric.type == MetricType.CUSTOM for metric in metrics):
            raise ValueError("PythonEnvironmentSystem does not support custom metrics.")

        model = model_config.create_model()
        metrics_res = {}
        for original_metric in metrics:
            metric = OliveEvaluator.generate_metric_user_config_with_model_io(original_metric, model)
            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = self.evaluate_accuracy(model, data_root, metric, accelerator)
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = self.evaluate_latency(model, data_root, metric, accelerator)
        return flatten_metric_result(metrics_res)

    def evaluate_accuracy(
        self, model: ONNXModel, data_root: str, metric: Metric, accelerator: AcceleratorSpec
    ) -> float:
        """
        Evaluate the accuracy of the model.
        """
        dataloader, _, post_func = OliveEvaluator.get_user_config(model.framework, data_root, metric)

        preds = []
        targets = []
        logits = []
        logits_dict = collections.defaultdict(list)
        inference_settings = self.get_inference_settings(model, metric, accelerator)
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
                targets.append(labels.cpu())
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
            is_single_tensor_output = len(output_names) == 1
            for idx in range(num_batches):
                output_path = output_dir / f"output_{idx}.npy"
                output = np.load(output_path)
                if is_single_tensor_output:
                    output = torch.Tensor(output[0])
                else:
                    # convert to dict of torch tensor
                    output = {name: torch.Tensor(output[i]) for i, name in enumerate(output_names)}
                if post_func:
                    output = post_func(output)
                preds.append(output.cpu())
                if is_single_tensor_output:
                    logits.append(output.cpu())
                else:
                    for k in output_names:
                        logits_dict[k].append(output[k].cpu())
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            if is_single_tensor_output:
                logits = torch.cat(logits, dim=0)
            else:
                logits = {k: torch.cat(logits[k], dim=0) for k in output_names}

        model_output = OliveModelOutput(preds, logits)
        return OliveEvaluator.compute_accuracy(metric, model_output, targets)

    def evaluate_latency(self, model: ONNXModel, data_root: str, metric: Metric, accelerator: AcceleratorSpec) -> float:
        """
        Evaluate the latency of the model.
        """
        dataloader, _, _ = OliveEvaluator.get_user_config(model.framework, data_root, metric)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)

        latencies = []
        inference_settings = self.get_inference_settings(model, metric, accelerator)
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

    def get_inference_settings(self, model: ONNXModel, metric: Metric, accelerator: AcceleratorSpec) -> Dict[str, Any]:
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
            execution_providers = (
                accelerator.execution_provider if accelerator else self.get_default_execution_provider(model)
            )
            inference_settings["execution_provider"] = execution_providers

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

    def install_requirements(self, accelerator: AcceleratorSpec):
        """
        Install required packages.
        """
        # install common packages
        common_requirements_file = Path(__file__).parent.resolve() / "common_requirements.txt"
        run_subprocess(
            f"pip install --cache-dir {self.environ['TMPDIR']} -r {common_requirements_file}",
            env=self.environ,
            check=True,
        )

        # install onnxruntime package
        onnxruntime_package = get_package_name(accelerator.execution_provider)
        run_subprocess(
            f"pip install --cache-dir {self.environ['TMPDIR']} {onnxruntime_package}",
            env=self.environ,
            check=True,
        )

        # install user requirements
        if self.config.requirements_file:
            run_subprocess(
                f"pip install --cache-dir {self.environ['TMPDIR']} -r {self.config.requirements_file}",
                env=self.environ,
                check=True,
            )

    def remove(self):
        import shutil

        vitual_env_path = Path(self.config.python_environment_path).resolve().parent

        try:
            shutil.rmtree(vitual_env_path)
            logger.info("Virtual environment '{}' removed.".format(vitual_env_path))
        except FileNotFoundError:
            pass

        if platform.system() == "Linux":
            try:
                shutil.rmtree(self.environ["TMPDIR"])
                logger.info("Temporary directory '{}' removed.".format(self.environ["TMPDIR"]))
            except FileNotFoundError:
                pass
