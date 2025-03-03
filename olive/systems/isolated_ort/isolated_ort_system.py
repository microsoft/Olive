# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import numpy as np

from olive.common.utils import format_data, load_weights, run_subprocess, save_weights
from olive.evaluator.metric import get_latency_config_from_metric
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveModelOutput, OnnxEvaluatorMixin, _OliveEvaluator
from olive.evaluator.registry import Registry
from olive.hardware import Device
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import IsolatedORTTargetUserConfig
from olive.systems.utils import create_new_environ, run_available_providers_runner

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from olive.evaluator.metric import Metric
    from olive.evaluator.metric_result import MetricResult
    from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ModelConfig, ONNXModelHandler
    from olive.passes.olive_pass import Pass

logger = logging.getLogger(__name__)


class IsolatedORTSystem(OliveSystem):
    system_type = SystemType.IsolatedORT

    def __init__(
        self,
        python_environment_path: Union[Path, str] = None,
        environment_variables: Dict[str, str] = None,
        prepend_to_path: List[str] = None,
        accelerators: List[AcceleratorConfig] = None,
        hf_token: bool = None,
    ):
        if python_environment_path is None:
            raise ValueError("python_environment_path is required for PythonEnvironmentSystem.")

        super().__init__(accelerators=accelerators, hf_token=hf_token)
        self.config = IsolatedORTTargetUserConfig(**locals())
        self.environ = create_new_environ(
            python_environment_path=python_environment_path,
            environment_variables=environment_variables,
            prepend_to_path=prepend_to_path,
        )

        # available eps. This will be populated the first time self.get_supported_execution_providers() is called.
        # used for caching the available eps
        self.available_eps = None

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: "ModelConfig",
        output_model_path: str,
    ) -> "ModelConfig":
        """Run the pass on the model."""
        logger.warning("IsolatedORTSystem does not support running passes.")
        raise NotImplementedError

    def evaluate_model(
        self, model_config: "ModelConfig", evaluator_config: "OliveEvaluatorConfig", accelerator: "AcceleratorSpec"
    ) -> "MetricResult":
        """Evaluate the model."""
        # only onnx model handler is supported
        if not model_config.type.lower() == "onnxmodel":
            raise ValueError(f"IsolatedORTSystem only supports evaluation for ONNXModel, got {model_config.type}")

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        model = model_config.create_model()
        evaluator = IsolatedORTEvaluator(self.environ)
        return evaluator.evaluate(
            model, evaluator_config.metrics, device=device, execution_providers=execution_providers
        )

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        if self.available_eps:
            return self.available_eps

        self.available_eps = run_available_providers_runner(self.environ)
        return self.available_eps

    def remove(self):
        raise NotImplementedError("ORT inference system does not support system removal")


@Registry.register("ort_inference")
@Registry.register("IsolatedORTEvaluator")
class IsolatedORTEvaluator(_OliveEvaluator, OnnxEvaluatorMixin):
    def __init__(self, environ: Dict[str, str]):
        super().__init__()

        assert environ, "environ should not be None"
        self.environ = environ
        self.inference_runner_path = Path(__file__).parent.resolve() / "inference_runner.py"
        self.executable = shutil.which("python", path=self.environ["PATH"])

    @classmethod
    def _get_common_config(
        cls, model: "ONNXModelHandler", metric: "Metric", device: Device, execution_providers: Union[str, List[str]]
    ) -> Dict:
        inference_settings = cls.get_inference_settings(metric, model)
        inference_settings = model.merge_inference_settings(inference_settings, execution_providers)
        return {
            "inference_settings": inference_settings,
            "use_ort_extensions": model.use_ort_extensions,
            "io_bind": cls.io_bind_enabled(metric, model.inference_settings),
            "device": str(device),
            "share_kv_buffer": metric.user_config.shared_kv_buffer,
            "use_fp16": any(v == "float16" for v in model.io_config["input_types"]),
        }

    def _run_inference(self, **kwargs):
        """Run inference using the inference runner.

        :param kwargs: arguments to be passed to the inference runner
        """
        command = [self.executable, str(self.inference_runner_path)]
        for key, value in kwargs.items():
            if not value:
                continue
            command.extend([f"--{key}", str(value)])
        run_subprocess(command, self.environ, check=True)

    def _inference(
        self,
        model: "ONNXModelHandler",
        metric: "Metric",
        dataloader: "Dataset",
        post_func: Callable = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
        import torch

        inference_config = self._get_common_config(model, metric, device, execution_providers)
        inference_config["mode"] = "inference"

        io_config = model.io_config

        preds = []
        targets = []
        logits = []
        logits_dict = collections.defaultdict(list)
        output_names = io_config["output_names"]
        is_single_tensor_output = len(output_names) == 1
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            # create input and output dir
            input_dir = temp_dir_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir = temp_dir_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            num_batches = 0
            for idx, (input_data, labels) in enumerate(dataloader):
                # save input data
                np.savez(input_dir / f"input_{idx}.npz", **format_data(input_data, io_config))
                # save labels
                targets.append(labels.cpu())
                num_batches += 1

            # external initializers or inputs can be any format, so we need to convert them to numpy
            external_inits_inputs = {}
            for name in ["external_initializers_path", "constant_inputs_path"]:
                external_path = getattr(model, name)
                if not external_path:
                    continue

                new_path = temp_dir_path / external_path.name
                external_inits_inputs[name] = save_weights(load_weights(external_path), new_path, "numpy")

            inference_config["num_batches"] = num_batches
            # save inference config
            config_path = temp_dir_path / "config.json"
            with config_path.open("w") as f:
                json.dump(inference_config, f)
            logger.debug("Inference config: %s", inference_config)

            # run inference
            self._run_inference(
                config_path=config_path,
                model_path=model.model_path,
                input_dir=input_dir,
                output_dir=output_dir,
                **external_inits_inputs,
            )

            # load and process output
            for idx in range(num_batches):
                result = np.load(output_dir / f"output_{idx}.npy")
                if is_single_tensor_output:
                    result = torch.Tensor(result[0])
                else:
                    result = {name: torch.Tensor(result[i]) for i, name in enumerate(output_names)}
                outputs = post_func(result) if post_func else result
                # keep as numpy or torch arrays
                preds.append(outputs.cpu())
                if is_single_tensor_output:
                    logits.append(result.cpu())
                else:
                    for k in output_names:
                        logits_dict[k].append(result[k].cpu())

            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            if is_single_tensor_output:
                logits = torch.cat(logits, dim=0)
            else:
                logits = {k: torch.cat(logits[k], dim=0) for k in output_names}

        return OliveModelOutput(preds=preds, logits=logits), targets

    def _evaluate_accuracy(
        self,
        model: "ONNXModelHandler",
        metric: "Metric",
        dataloader: "Dataset",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> "MetricResult":
        inference_output, targets = self._inference(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    def _evaluate_raw_latency(
        self,
        model: "ONNXModelHandler",
        metric: "Metric",
        dataloader: "Dataset",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        """For given repeat_test_num, return a list of latencies(ms)."""
        inference_config = self._get_common_config(model, metric, device, execution_providers)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        inference_config.update(
            {
                "mode": "latency",
                "warmup_num": warmup_num,
                "repeat_test_num": repeat_test_num,
                "sleep_num": sleep_num,
            }
        )

        io_config = model.io_config

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            # create input and output dir
            input_dir = temp_dir_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir = temp_dir_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # save input data
            np.savez(input_dir / "input.npz", **format_data(next(iter(dataloader))[0], io_config))

            # save inference config
            config_path = temp_dir_path / "config.json"
            with config_path.open("w") as f:
                json.dump(inference_config, f)

            # run inference
            self._run_inference(
                config_path=config_path,
                model_path=model.model_path,
                input_dir=input_dir,
                output_dir=output_dir,
                external_initializers_path=model.external_initializers_path,
                constant_inputs_path=model.constant_inputs_path,
            )

            # load output
            return np.load(output_dir / "output.npy").tolist()
