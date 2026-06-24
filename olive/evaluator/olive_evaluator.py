# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Optional, Union

import numpy as np
import torch
from pydantic import Field, field_validator, model_validator

from olive.common.config_utils import NestedConfig, validate_config
from olive.common.import_lib import import_user_module
from olive.common.ort_inference import OrtInferenceSession, prepare_io_bindings
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import format_data, load_weights, tensor_data_to_device
from olive.constants import Framework
from olive.data.config import DataConfig
from olive.data.container.dummy_data_container import TRANSFORMER_DUMMY_DATA_CONTAINER
from olive.data.template import dummy_data_config_template
from olive.evaluator.metric import (
    AccuracySubType,
    LatencySubType,
    Metric,
    MetricType,
    SizeOnDiskSubType,
    ThroughputSubType,
    get_latency_config_from_metric,
)
from olive.evaluator.metric_backend import MetricBackend
from olive.evaluator.metric_result import MetricResult, SubMetricResult, flatten_metric_result, joint_metric_key
from olive.evaluator.registry import Registry
from olive.hardware import Device
from olive.model import DistributedOnnxModelHandler, ONNXModelHandler, PyTorchModelHandler
from olive.model.config.io_config import is_io_config_static
from olive.model.handler.hf import HfModelHandler
from olive.model.utils.onnx_utils import dump_tuning_result
from olive.platform_sdk.qualcomm.utils.data_loader import FileListCommonDataLoader, FileListDataLoader

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from olive.model import OliveModelHandler, OpenVINOModelHandler, QNNModelHandler

logger = logging.getLogger(__name__)

# pylint: disable=useless-parent-delegation


class OliveModelOutput(NamedTuple):
    preds: Any
    logits: Any


# Text-based accuracy sub-types that work with string predictions/targets
_TEXT_BASED_ACCURACY_SUBTYPES = {AccuracySubType.WER, AccuracySubType.RTFX}
_VISION_ACCURACY_SUBTYPES = {
    AccuracySubType.EXACT_MATCH,
    AccuracySubType.RELAXED_ACCURACY,
    AccuracySubType.WORD_SORT_RATIO,
}

# Task-to-metric validation: maps data task types to their allowed vision metrics.
# Metrics are aligned with standard public vision benchmarks:
#   - vision-vqa (exact_match): AI2D, ScienceQA, TextVQA, MathVista, MMMU, InterGPS
#   - vision-chart-qa (relaxed_accuracy): ChartQA (±5% numeric tolerance)
#   - vision-ocr (word_sort_ratio): OCR (word-level overlap)
_VISION_TASK_METRIC_MAP = {
    "vision-vqa": {AccuracySubType.EXACT_MATCH},
    "vision-chart-qa": {AccuracySubType.RELAXED_ACCURACY},
    "vision-ocr": {AccuracySubType.WORD_SORT_RATIO},
}


def _is_text_based_metric(metric: "Metric") -> bool:
    """Check if metric uses text-based accuracy sub-types (WER, RTFx).

    Raises ValueError if text-based and tensor-based sub-types are mixed,
    as they require different inference paths.
    """
    if metric.type != MetricType.ACCURACY:
        return False
    text_based = [sub.name in _TEXT_BASED_ACCURACY_SUBTYPES for sub in metric.sub_types]
    if any(text_based) and not all(text_based):
        raise ValueError(
            "Cannot mix text-based accuracy sub-types (WER, RTFx) with tensor-based sub-types "
            "(accuracy_score, f1_score, etc.) in the same metric. Please define them as separate metrics."
        )
    return all(text_based)


def _is_vision_metric(metric: "Metric") -> bool:
    """Check if metric uses vision accuracy sub-types (exact_match, relaxed_accuracy, word_sort_ratio).

    Raises ValueError if vision sub-types are mixed with non-vision sub-types,
    as they require different inference paths.
    """
    if metric.type != MetricType.ACCURACY:
        return False
    vision_based = [sub.name in _VISION_ACCURACY_SUBTYPES for sub in metric.sub_types]
    if any(vision_based) and not all(vision_based):
        raise ValueError(
            "Cannot mix vision accuracy sub-types (exact_match, relaxed_accuracy, word_sort_ratio) "
            "with other sub-types in the same metric. Please define them as separate metrics."
        )
    return all(vision_based)


def _validate_vision_task_metric(metric: "Metric") -> None:
    """Validate that the vision metric sub-types are compatible with the data task type.

    Raises ValueError if the metric is not compatible with the task.
    """
    if not _is_vision_metric(metric):
        return

    task_type = None
    if metric.data_config:
        # Extract task from pre_process_data_config params, which is how HuggingfaceContainer
        # maps task types (e.g., "vision-vqa", "vision-chart-qa", "vision-ocr") to components.
        pre_process_config = metric.data_config.pre_process_data_config
        if pre_process_config:
            if pre_process_config.params:
                task_type = pre_process_config.params.get("task")
            # Also try to infer task from the component type name if params don't specify it
            if task_type is None and pre_process_config.type == "vision_vqa_pre_process":
                # Default component is used but task param is missing; skip validation
                # since we can't determine which specific vision task is intended
                return

    if task_type is None:
        # No task type specified, allow any vision metric
        return

    allowed_metrics = _VISION_TASK_METRIC_MAP.get(task_type)
    if allowed_metrics is None:
        raise ValueError(
            f"Unknown vision task type '{task_type}'. Supported task types: {list(_VISION_TASK_METRIC_MAP.keys())}."
        )

    for sub in metric.sub_types:
        if sub.name not in allowed_metrics:
            raise ValueError(
                f"Metric sub-type '{sub.name}' is not compatible with task type '{task_type}'. "
                f"Allowed metrics for '{task_type}': {[m.value for m in allowed_metrics]}."
            )


class OliveEvaluator(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def unpack_batch_for_accuracy(batch):
        """Unpack batch for accuracy evaluation. Requires (data, label) format."""
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError(
                "Accuracy evaluation requires dataset with labels. "
                "Please use ClassificationDataset which returns (data, label)."
            )
        return batch[0], batch[1]

    @staticmethod
    def extract_input_data(batch):
        """Extract input data from batch, ignoring labels if present."""
        return batch[0] if isinstance(batch, (tuple, list)) and len(batch) == 2 else batch

    @abstractmethod
    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: list[Metric],
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        raise NotImplementedError

    @staticmethod
    def generate_metric_user_config_with_model_io(metric: Metric, model: "OliveModelHandler"):
        # if the io_config is not specified in the data config, use the one in the model
        # should not change the original metric object which is created from config jsons
        # otherwise, if affects hashing + caching of the olive restoring.
        metric = deepcopy(metric)
        if metric.data_config:
            return metric

        if metric.type != MetricType.LATENCY:
            return metric

        io_config = model.io_config
        if not io_config:
            return metric

        if not is_io_config_static(io_config):
            # since Olive will not save the pytorch model's io_config to olive onnx model
            # we cannot generate dummy data for the onnx model if this model has dynamic input shapes
            # TODO(trajep): try to get static input shapes from onnx model.
            # If so, we can move the dataloader for latency measurement.
            logger.error(
                "Model input shapes are not static. Cannot use inferred input shapes for creating dummy data. This will"
                " cause an error when creating dummy data for tuning."
            )

        metric.data_config = dummy_data_config_template(
            io_config.get("input_shapes"), io_config.get("input_names"), io_config.get("input_types")
        )
        metric.data_config = validate_config(metric.data_config, DataConfig)
        return metric

    @staticmethod
    def get_user_config(framework: Framework, metric: Metric):
        dataloader = None
        eval_func = None
        post_func = None

        # load the evaluate function
        # priority: evaluate_func > metric_func
        if metric.type == MetricType.CUSTOM:
            if not metric.user_config:
                raise ValueError("user_config is required for CUSTOM metric type")

            evaluate_func = getattr(metric.user_config, "evaluate_func", None)
            kwargs = getattr(metric.user_config, "evaluate_func_kwargs", None) or {}
            if not evaluate_func:
                evaluate_func = getattr(metric.user_config, "metric_func", None)
                kwargs = getattr(metric.user_config, "metric_func_kwargs", None) or {}

            if not evaluate_func:
                raise ValueError("evaluate_func or metric_func is not specified in the metric config")

            user_module = UserModuleLoader(metric.user_config.user_script, metric.user_config.script_dir)
            eval_func = user_module.load_object(evaluate_func)
            if kwargs:
                eval_func = partial(eval_func, **kwargs)

        # get dataloader and/or post processing function from data_config if not specified in the metric config
        if metric.data_config:
            if metric.data_config.type in TRANSFORMER_DUMMY_DATA_CONTAINER:
                metric.data_config.load_dataset_config.params["model_framework"] = framework

            dc = metric.data_config.to_data_container()
            dataloader = dc.create_dataloader()
            post_func = dc.config.post_process

        return dataloader, eval_func, post_func

    @staticmethod
    def compute_accuracy(metric: Metric, model_outputs: Union[tuple, NamedTuple], targets: Any) -> MetricResult:
        """Compute accuracy metrics."""
        evaluate_backend_cls = MetricBackend.registry[metric.backend]
        return evaluate_backend_cls().measure(model_outputs, targets, metric)

    @staticmethod
    def latency_helper(latencies) -> dict:
        return {
            LatencySubType.AVG: round(sum(latencies) / len(latencies) * 1000, 5),
            LatencySubType.MAX: round(max(latencies) * 1000, 5),
            LatencySubType.MIN: round(min(latencies) * 1000, 5),
            LatencySubType.P50: round(np.percentile(latencies, 50) * 1000, 5),
            LatencySubType.P75: round(np.percentile(latencies, 75) * 1000, 5),
            LatencySubType.P90: round(np.percentile(latencies, 90) * 1000, 5),
            LatencySubType.P95: round(np.percentile(latencies, 95) * 1000, 5),
            LatencySubType.P99: round(np.percentile(latencies, 99) * 1000, 5),
            LatencySubType.P999: round(np.percentile(latencies, 99.9) * 1000, 5),
        }

    @staticmethod
    def compute_latency(metric: Metric, latencies: Any) -> MetricResult:
        """Compute latency metrics."""
        latency_metrics = OliveEvaluator.latency_helper(latencies)
        metric_res = {}
        for sub_type in metric.sub_types:
            metric_res[sub_type.name] = SubMetricResult(
                value=latency_metrics[sub_type.name],
                priority=sub_type.priority,
                higher_is_better=sub_type.higher_is_better,
            )
        return MetricResult.model_validate(metric_res)

    @staticmethod
    def compute_throughput(metric: Metric, latencies: Any) -> MetricResult:
        """Compute throughput metrics."""
        latency_metrics = OliveEvaluator.latency_helper(latencies)
        metric_res = {}
        batch_size = metric.data_config.dataloader_params.get("batch_size", 1) if metric.data_config else 1
        for sub_type in metric.sub_types:
            if sub_type.name == ThroughputSubType.MIN:
                latency_sub_type_name = LatencySubType.MAX
            elif sub_type.name == ThroughputSubType.MAX:
                latency_sub_type_name = LatencySubType.MIN
            else:
                latency_sub_type_name = LatencySubType(sub_type.name)
            metric_res[sub_type.name] = SubMetricResult(
                # per second, so multiply by 1000
                value=round(batch_size / latency_metrics[latency_sub_type_name] * 1000, 5),
                priority=sub_type.priority,
                higher_is_better=sub_type.higher_is_better,
            )
        return MetricResult.model_validate(metric_res)


class _OliveEvaluator(OliveEvaluator):
    @staticmethod
    def device_string_to_torch_device(device: Device):
        return torch.device("cuda") if device == Device.GPU and torch.cuda.is_available() else torch.device("cpu")

    @classmethod
    def io_bind_enabled(cls, metric: Metric, inference_settings: dict) -> bool:
        if metric.user_config and metric.user_config.io_bind:
            return True

        return inference_settings and inference_settings.get("io_bind")

    @abstractmethod
    def _inference(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_accuracy(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_raw_latency(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        """For given repeat_test_num, return a list of latencies(ms)."""
        raise NotImplementedError

    def _evaluate_latency(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        latencies = self._evaluate_raw_latency(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_latency(metric, latencies)

    def _evaluate_throughput(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        latencies = self._evaluate_raw_latency(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_throughput(metric, latencies)

    def _evaluate_size_on_disk(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        return MetricResult.model_validate(
            {SizeOnDiskSubType.BYTES.value: {"value": model.size_on_disk, "priority": -1, "higher_is_better": False}}
        )

    def _evaluate_custom(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        eval_func,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers=None,
    ) -> MetricResult:
        raw_res = None
        if metric.user_config and metric.user_config.evaluate_func:
            raw_res = eval_func(model, device, execution_providers)
        else:
            inference_output, targets = self._inference(
                model, metric, dataloader, post_func, device, execution_providers
            )
            raw_res = eval_func(inference_output, targets)

        metric_res = {}
        for sub_type in metric.sub_types:
            if isinstance(raw_res, Number):
                assert len(metric.sub_types) == 1, "Only one sub type is allowed for single value custom metric"
                metric_res[sub_type.name] = SubMetricResult(
                    value=raw_res, priority=sub_type.priority, higher_is_better=sub_type.higher_is_better
                )
            elif isinstance(raw_res, dict):
                assert sub_type.name in raw_res, f"Custom metric {sub_type.name} is not in the result"
                metric_res[sub_type.name] = SubMetricResult(
                    value=raw_res[sub_type.name],
                    priority=sub_type.priority,
                    higher_is_better=sub_type.higher_is_better,
                )
        return MetricResult.model_validate(metric_res)

    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: list[Metric],
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        metrics_res = {}
        for original_metric in metrics:
            # use model io_config if user does not specify input_names and input_shapes
            metric = OliveEvaluator.generate_metric_user_config_with_model_io(original_metric, model)
            dataloader, eval_func, post_func = OliveEvaluator.get_user_config(model.framework, metric)
            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = self._evaluate_accuracy(
                    model, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = self._evaluate_latency(
                    model, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.THROUGHPUT:
                metrics_res[metric.name] = self._evaluate_throughput(
                    model, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.SIZE_ON_DISK:
                metrics_res[metric.name] = self._evaluate_size_on_disk(
                    model, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.CUSTOM:
                metrics_res[metric.name] = self._evaluate_custom(
                    model, metric, dataloader, eval_func, post_func, device, execution_providers
                )
            else:
                raise TypeError(f"{metric.type} is not a supported metric type")
        return flatten_metric_result(metrics_res)


class OnnxEvaluatorMixin:
    @staticmethod
    def get_inference_settings(metric: Metric, model: ONNXModelHandler) -> dict[str, Any]:
        # user.config.inference_settings > model.inference_settings > default inference_settings
        # when user.config.inference_settings is None, the model.inference_settings
        # will be used in model.prepare_session(..)
        inference_settings = {}
        model_inference_settings = model.inference_settings
        if model_inference_settings:
            inference_settings.update(model_inference_settings)

        metric_inference_settings = metric.get_inference_settings(Framework.ONNX.lower())
        if metric_inference_settings:
            inference_settings.update(metric_inference_settings)

        return inference_settings


def _find_genai_config(model: ONNXModelHandler) -> Optional[Path]:
    """Find genai_config.json by searching upward from the ONNX file's parent directory.

    Returns the Path to genai_config.json if found, or None. Searches at most
    3 levels up to avoid traversing unrelated directories.
    """
    candidate = Path(model.model_path).parent
    for _ in range(3):
        genai_path = candidate / "genai_config.json"
        if genai_path.is_file():
            return genai_path
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return None


def _get_genai_model_dir(model: ONNXModelHandler) -> str:
    """Get the ORT GenAI model root directory (where genai_config.json lives).

    Falls back to the ONNX file's parent directory if genai_config.json is not found.
    """
    genai_config_path = _find_genai_config(model)
    if genai_config_path is not None:
        return str(genai_config_path.parent)
    return str(Path(model.model_path).parent)


@Registry.register(str(Framework.ONNX))
@Registry.register("OnnxEvaluator")
class OnnxEvaluator(_OliveEvaluator, OnnxEvaluatorMixin):
    @staticmethod
    def get_session_wrapper(
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        device: Device,
        execution_providers: list[str],
    ) -> tuple[OrtInferenceSession, dict[str, Any]]:
        """Get the session wrapper for the model."""
        # user.config.inference_settings > model.inference_settings > default inference_settings
        inference_settings = OnnxEvaluator.get_inference_settings(metric, model)
        session = model.prepare_session(
            inference_settings=inference_settings,
            device=device,
            execution_providers=execution_providers,
        )

        # prepare for io binding
        io_config = model.io_config
        io_bind = OnnxEvaluator.io_bind_enabled(metric, model.inference_settings)
        shared_kv_buffer = getattr(metric.user_config, "shared_kv_buffer", None) if metric.user_config else None
        use_fp16 = any(v == "float16" for v in io_config["input_types"])
        input_feed = None
        if io_bind and shared_kv_buffer and use_fp16:
            batch = next(iter(dataloader))
            input_data = OliveEvaluator.extract_input_data(batch)
            input_feed = format_data(input_data, io_config)

        # load constant inputs if any
        constant_inputs = None
        if model.constant_inputs_path:
            constant_inputs = format_data(load_weights(model.constant_inputs_path), io_config)

        # create session wrapper
        session_wrapper = OrtInferenceSession(
            session,
            io_bind=io_bind,
            device=device,
            shared_kv_buffer=shared_kv_buffer,
            use_fp16=use_fp16,
            input_feed=input_feed,
            constant_inputs=constant_inputs,
        )

        return session_wrapper, inference_settings

    def _inference(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        session, inference_settings = OnnxEvaluator.get_session_wrapper(
            model, metric, dataloader, device, execution_providers
        )
        io_config = model.io_config
        run_kwargs = metric.get_run_kwargs()

        preds = []
        targets = []
        logits = []
        logits_dict = collections.defaultdict(list)
        output_names = io_config["output_names"]
        is_single_tensor_output = len(output_names) == 1
        for batch in dataloader:
            input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            input_feed = format_data(input_data, io_config)
            result = model.run_session(session, input_feed, **run_kwargs)
            if is_single_tensor_output:
                result = torch.Tensor(result[0])
            else:
                # convert to dict of torch tensor
                result = {name: torch.Tensor(result[i]) for i, name in enumerate(output_names)}
            outputs = post_func(result) if post_func else result
            # keep as numpy or torch arrays
            preds.append(outputs.cpu())
            targets.append(labels.cpu())
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
            logits = {k: torch.cat(logits_dict[k], dim=0) for k in output_names}

        tuning_result_file = inference_settings.get("tuning_result_file")
        if tuning_result_file:
            dump_tuning_result(session.session, tuning_result_file)
        return OliveModelOutput(preds=preds, logits=logits), targets

    @staticmethod
    def _load_genai_config(model: ONNXModelHandler) -> Optional[dict]:
        """Load genai_config.json from the model directory, or return None if not found.

        Searches upward from the ONNX file's parent directory to support nested
        multi-component model layouts (e.g. ``models/decoder/model.onnx`` where
        ``genai_config.json`` lives at ``models/``).
        """
        genai_config_path = _find_genai_config(model)
        if genai_config_path is None:
            return None
        import json

        try:
            with genai_config_path.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in genai config file: {genai_config_path}") from e

    def _evaluate_onnx_accuracy(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        if _is_vision_metric(metric):
            _validate_vision_task_metric(metric)
            # Auto-detect genai vision model by checking for genai_config.json with vision field
            genai_cfg = self._load_genai_config(model)
            use_genai_vision = genai_cfg is not None and "vision" in genai_cfg.get("model", {})

            if use_genai_vision:
                inference_output, targets = self._inference_vision_genai(model, dataloader, device)
            else:
                inference_output, targets = self._inference_vision(
                    model, metric, dataloader, post_func, device, execution_providers
                )
        elif _is_text_based_metric(metric):
            # Auto-detect genai model by checking for genai_config.json
            genai_cfg = self._load_genai_config(model)
            if genai_cfg:
                model_type = genai_cfg.get("model", {}).get("type", "")

                if model_type == "whisper":
                    inference_output, targets = self._inference_text_genai(
                        model, metric, dataloader, device, execution_providers
                    )
                elif model_type == "nemotron_speech":
                    inference_output, targets = self._inference_text_genai_streaming(
                        model, metric, dataloader, device, execution_providers
                    )
                else:
                    raise ValueError(
                        f"Unsupported genai model type '{model_type}' for speech evaluation. "
                        f"Supported types: 'whisper' (offline), 'nemotron_speech' (streaming). "
                        f"For unsupported model types, use a custom evaluation script."
                    )
            else:
                inference_output, targets = self._inference_text(
                    model, metric, dataloader, post_func, device, execution_providers
                )
        else:
            inference_output, targets = self._inference(
                model, metric, dataloader, post_func, device, execution_providers
            )
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    def _inference_text(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        """Text-based inference for speech/ASR metrics (WER, RTFx).

        The post_func must return a list of predicted text strings per batch.
        Labels from the dataloader must be a list of reference text strings.
        Tracks total inference time and audio duration for RTFx computation.
        """
        session, inference_settings = OnnxEvaluator.get_session_wrapper(
            model, metric, dataloader, device, execution_providers
        )
        io_config = model.io_config
        run_kwargs = metric.get_run_kwargs()

        all_preds = []
        all_targets = []
        total_audio_duration = 0.0
        total_inference_time = 0.0
        output_names = io_config["output_names"]
        is_single_tensor_output = len(output_names) == 1
        sample_rate = (
            metric.data_config.pre_process_data_config.params.get("sample_rate", 16000)
            if (metric.data_config and metric.data_config.pre_process_data_config)
            else 16000
        )

        for batch in dataloader:
            input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            # Track audio duration from input data
            if isinstance(input_data, (np.ndarray, torch.Tensor)):
                audio_samples = input_data.shape[-1] if len(input_data.shape) > 1 else input_data.shape[0]
                total_audio_duration += audio_samples / sample_rate
            elif isinstance(input_data, dict):
                for v in input_data.values():
                    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 1:
                        total_audio_duration += v.shape[-1] / sample_rate
                        break

            input_feed = format_data(input_data, io_config)
            start_time = time.perf_counter()
            result = model.run_session(session, input_feed, **run_kwargs)
            if is_single_tensor_output:
                result = torch.from_numpy(result[0]) if hasattr(result[0], "__array__") else torch.tensor(result[0])
            else:
                result = {
                    name: torch.from_numpy(result[i]) if hasattr(result[i], "__array__") else torch.tensor(result[i])
                    for i, name in enumerate(output_names)
                }
            # post_func must decode model output to text strings
            outputs = post_func(result) if post_func else result
            total_inference_time += time.perf_counter() - start_time

            if isinstance(outputs, str):
                all_preds.append(outputs)
            elif isinstance(outputs, (list, tuple)):
                if not outputs:
                    continue
                if not isinstance(outputs[0], str):
                    raise ValueError(
                        f"post_func must return str or list[str] for text-based metrics (WER), "
                        f"but got list of {type(outputs[0]).__name__}. "
                        f"Ensure your post_func decodes model output to text."
                    )
                all_preds.extend(outputs)
            else:
                raise ValueError(
                    f"post_func must return str or list[str] for text-based metrics (WER), "
                    f"but got {type(outputs).__name__}. "
                    f"Ensure your post_func decodes model output to text."
                )
            # labels should be reference text strings
            if isinstance(labels, (list, tuple)):
                all_targets.extend(labels)
            else:
                all_targets.append(labels)

        tuning_result_file = inference_settings.get("tuning_result_file")
        if tuning_result_file:
            dump_tuning_result(session.session, tuning_result_file)

        # Store timing metadata for RTFx computation
        timing_metadata = {
            "total_audio_duration": total_audio_duration,
            "total_inference_time": total_inference_time,
        }
        return OliveModelOutput(preds=all_preds, logits=timing_metadata), all_targets

    def _inference_vision(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        """Vision-based inference for VQA/OCR metrics (exact_match, relaxed_accuracy, word_sort_ratio).

        The post_func must return predicted answer strings per batch.
        Labels from the dataloader must be reference answer strings.
        """
        session, inference_settings = OnnxEvaluator.get_session_wrapper(
            model, metric, dataloader, device, execution_providers
        )
        io_config = model.io_config
        run_kwargs = metric.get_run_kwargs()

        all_preds = []
        all_targets = []
        output_names = io_config["output_names"]
        is_single_tensor_output = len(output_names) == 1

        # Note: This assumes the model produces the full answer in a single forward pass
        # (e.g., classification-style VQA models). For autoregressive generation models,
        # use the PyTorch evaluator with a generation loop in post_func instead.
        for batch in dataloader:
            input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            input_feed = format_data(input_data, io_config)
            result = model.run_session(session, input_feed, **run_kwargs)
            if is_single_tensor_output:
                result = torch.from_numpy(result[0]) if hasattr(result[0], "__array__") else torch.tensor(result[0])
            else:
                result = {
                    name: torch.from_numpy(result[i]) if hasattr(result[i], "__array__") else torch.tensor(result[i])
                    for i, name in enumerate(output_names)
                }
            # post_func must decode model output to answer strings
            outputs = post_func(result) if post_func else result
            if isinstance(outputs, str):
                all_preds.append(outputs)
            elif isinstance(outputs, (list, tuple)):
                if not outputs:
                    continue
                if not isinstance(outputs[0], str):
                    raise ValueError(
                        f"post_func must return str or list[str] for vision metrics, "
                        f"but got list of {type(outputs[0]).__name__}. "
                        f"Ensure your post_func decodes model output to answer text."
                    )
                all_preds.extend(outputs)
            else:
                raise ValueError(
                    f"post_func must return str or list[str] for vision metrics, "
                    f"but got {type(outputs).__name__}. "
                    f"Ensure your post_func decodes model output to answer text."
                )
            # labels should be reference answer strings
            if isinstance(labels, (list, tuple)):
                all_targets.extend(labels)
            else:
                all_targets.append(labels)

        tuning_result_file = inference_settings.get("tuning_result_file")
        if tuning_result_file:
            dump_tuning_result(session.session, tuning_result_file)

        return OliveModelOutput(preds=all_preds, logits=None), all_targets

    def _inference_vision_genai(
        self,
        model: ONNXModelHandler,
        dataloader: "DataLoader",
        device: Device = Device.CPU,
    ) -> tuple[OliveModelOutput, Any]:
        """Vision-based inference for VQA/OCR metrics using onnxruntime-genai.

        Auto-detected when the model directory contains genai_config.json with a vision field.
        Uses og.Model with multimodal processor for vision-language models (e.g., Qwen3-VL).
        The dataloader must yield (input_dict, labels) where input_dict contains
        'image' (PIL Image) and 'question' (str), and labels are reference answer strings.

        Note: GPU/CPU selection is driven by the `device` parameter. onnxruntime-genai uses
        short provider names internally (e.g., "cuda") which differ from ORT-style EP names.
        """
        try:
            import onnxruntime_genai as og
        except ImportError as e:
            raise ImportError(
                "onnxruntime-genai is required for genai-based vision evaluation. "
                "Install it with: pip install onnxruntime-genai"
            ) from e

        import json
        import re
        import tempfile

        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required for vision evaluation. Install it with: pip install Pillow") from e

        model_dir = _get_genai_model_dir(model)

        # Default max_length; can be overridden per-sample from the data config.
        default_max_length = 4096

        # Build og.Model with appropriate execution provider
        # Note: onnxruntime-genai uses CPU by default when no provider is appended.
        # Only non-CPU providers need to be explicitly added using short names (e.g., "cuda").
        # This follows the same pattern as _inference_text_genai and _inference_text_genai_streaming.
        config = og.Config(model_dir)
        config.clear_providers()
        if device == Device.GPU:
            config.append_provider("cuda")
        og_model = og.Model(config)
        processor = og_model.create_multimodal_processor()
        tokenizer = og.Tokenizer(og_model)

        all_preds = []
        all_targets = []

        # Use a temporary directory for image files to avoid per-file create/delete overhead
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_img_path = Path(tmp_dir) / "input.png"

            sample_idx = 0
            for batch in dataloader:
                input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)

                # input_data is a dict with 'image' (PIL) and 'question' (str)
                # or a list of such dicts for batch_size > 1
                items = [input_data] if isinstance(input_data, dict) else input_data

                for item in items:
                    pil_image = item.get("image")
                    question = item.get("question", "")
                    sys_prompt = item.get("system_prompt", "")
                    num_choices = item.get("num_choices", 0)
                    max_length = item.get("max_length", default_max_length)

                    if pil_image is None:
                        # Append empty pred to maintain alignment with targets
                        all_preds.append("")
                        sample_idx += 1
                        continue

                    try:
                        # Ensure PIL Image
                        if not isinstance(pil_image, Image.Image):
                            with Image.open(pil_image) as img:
                                pil_image = img.convert("RGB")

                        # Build chat messages for the vision-language model
                        messages = []
                        if sys_prompt:
                            messages.append({"role": "system", "content": sys_prompt})
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": question},
                                ],
                            }
                        )
                        messages_json = json.dumps(messages)

                        # Save image to temp file for og.Images (reuse same path to minimize I/O)
                        pil_image.save(str(tmp_img_path), format="PNG")
                        images = og.Images.open(str(tmp_img_path))

                        prompt = tokenizer.apply_chat_template(messages_json, add_generation_prompt=True)
                        inputs = processor(prompt, images=images)

                        # Remove audio_features if present but not needed (vision-only inference)
                        # to avoid "Model output was not found: audio_features" errors
                        if "audio_features" in inputs:
                            del inputs["audio_features"]

                        params = og.GeneratorParams(og_model)
                        params.set_search_options(max_length=max_length, do_sample=False)

                        generator = og.Generator(og_model, params)
                        generator.set_inputs(inputs)

                        tokens = []
                        while not generator.is_done():
                            generator.generate_next_token()
                            tokens.append(generator.get_next_tokens()[0])
                        del generator

                        pred = tokenizer.decode(tokens).strip()
                    except Exception as e:
                        logger.warning("Skipping sample %d due to error: %s", sample_idx, e)
                        pred = ""

                    sample_idx += 1

                    # For multiple-choice tasks, extract the answer digit from responses
                    # like "2", "The answer is 3", or "1. D" to match the expected answer format.
                    # Only enabled when num_choices is between 1 and 9 (single-digit options).
                    if 1 <= num_choices <= 9 and pred:
                        pattern = rf"\b([1-{num_choices}])\b"
                        num_match = re.search(pattern, pred)
                        if num_match:
                            pred = num_match.group(1)
                        else:
                            # Fallback: find any single digit in the valid range
                            valid_digits = {str(d) for d in range(1, num_choices + 1)}
                            for ch in pred:
                                if ch in valid_digits:
                                    pred = ch
                                    break
                    all_preds.append(pred)

                # Collect reference texts (aligned with preds including empty ones for None images)
                if isinstance(labels, (list, tuple)):
                    all_targets.extend(labels)
                else:
                    all_targets.append(labels)

        del og_model

        return OliveModelOutput(preds=all_preds, logits=None), all_targets

    def _inference_text_genai(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        """Text-based inference for speech/ASR metrics using onnxruntime-genai.

        Auto-detected when the model directory contains genai_config.json.
        Uses og.Model with multimodal processor for Whisper-style models.
        Automatically chunks audio longer than 30 seconds.
        """
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise ImportError(
                "onnxruntime-genai is required for genai-based speech evaluation. "
                "Install it with: pip install onnxruntime-genai"
            ) from None

        import io
        import json

        import soundfile as sf

        model_dir = _get_genai_model_dir(model)

        # Read genai_config to determine model properties
        with (Path(model_dir) / "genai_config.json").open() as f:
            genai_config = json.load(f)

        # Build og.Model with appropriate execution provider
        config = og.Config(model_dir)
        config.clear_providers()
        if device == Device.GPU:
            config.append_provider("cuda")
        og_model = og.Model(config)
        processor = og_model.create_multimodal_processor()

        # Determine decoder prompt tokens from model config
        # English-only models (vocab_size=51864) use shorter prompt
        vocab_size = genai_config.get("model", {}).get("vocab_size", 51865)
        is_english_only = vocab_size == 51864
        if is_english_only:
            decoder_prompt_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
        else:
            decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]

        sample_rate = (
            metric.data_config.pre_process_data_config.params.get("sample_rate", 16000)
            if (metric.data_config and metric.data_config.pre_process_data_config)
            else 16000
        )
        max_length = genai_config.get("search", {}).get("max_length", 448)

        # Whisper encoder supports max 30s (3000 mel frames)
        max_chunk_seconds = 30
        max_chunk_samples = max_chunk_seconds * sample_rate

        prompt = "".join(decoder_prompt_tokens)

        def _transcribe_chunks(audio_arr: np.ndarray, genai_model) -> str:
            """Transcribe a single audio array, chunking if longer than 30s."""
            if len(audio_arr) <= max_chunk_samples:
                chunks = [audio_arr]
            else:
                # Split into non-overlapping 30s chunks
                chunks = []
                for start in range(0, len(audio_arr), max_chunk_samples):
                    chunks.append(audio_arr[start : start + max_chunk_samples])

            transcriptions = []
            for chunk in chunks:
                buffer = io.BytesIO()
                sf.write(buffer, chunk, samplerate=sample_rate, format="WAV")
                audios = og.Audios.open_bytes(buffer.getvalue())
                inputs = processor([prompt], audios=audios)

                params = og.GeneratorParams(genai_model)
                params.set_search_options(do_sample=False, max_length=max_length, min_length=0, batch_size=1)

                generator = og.Generator(genai_model, params)
                generator.set_inputs(inputs)

                while not generator.is_done():
                    generator.generate_next_token()

                tokens = generator.get_sequence(0)
                transcriptions.append(processor.decode(tokens).strip())

            return " ".join(transcriptions)

        all_preds = []
        all_targets = []
        total_audio_duration = 0.0
        total_inference_time = 0.0

        for batch in dataloader:
            input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)

            # Convert input to list of audio arrays
            audio_arrays = []
            if isinstance(input_data, (np.ndarray, torch.Tensor)):
                arr = np.array(input_data) if isinstance(input_data, torch.Tensor) else input_data
                if arr.ndim == 1:
                    audio_arrays = [arr]
                else:
                    audio_arrays = [arr[i] for i in range(arr.shape[0])]
            elif isinstance(input_data, list):
                audio_arrays = [np.array(a) if not isinstance(a, np.ndarray) else a for a in input_data]

            if not audio_arrays:
                continue

            start_time = time.perf_counter()
            for arr in audio_arrays:
                total_audio_duration += len(arr) / sample_rate
                transcription = _transcribe_chunks(arr, og_model)
                all_preds.append(transcription)
            total_inference_time += time.perf_counter() - start_time

            # Collect reference texts
            if isinstance(labels, (list, tuple)):
                all_targets.extend(labels)
            else:
                all_targets.append(labels)

        del og_model

        timing_metadata = {
            "total_audio_duration": total_audio_duration,
            "total_inference_time": total_inference_time,
        }
        return OliveModelOutput(preds=all_preds, logits=timing_metadata), all_targets

    def _inference_text_genai_streaming(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        """Text-based inference for streaming ASR models using onnxruntime-genai.

        Auto-detected when genai_config.json has model.type = "nemotron_speech".
        Uses og.StreamingProcessor for stateful chunked inference with silence padding
        for right-context flushing.
        """
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise ImportError(
                "onnxruntime-genai is required for genai-based speech evaluation. "
                "Install it with: pip install onnxruntime-genai"
            ) from None

        import json

        model_dir = _get_genai_model_dir(model)

        with (Path(model_dir) / "genai_config.json").open() as f:
            genai_config = json.load(f)

        sample_rate = genai_config["model"].get("sample_rate", 16000)
        chunk_samples = genai_config["model"].get("chunk_samples", 8960)

        # Build og.Model with appropriate execution provider
        config = og.Config(model_dir)
        config.clear_providers()
        if device == Device.GPU:
            config.append_provider("cuda")
        og_model = og.Model(config)
        tokenizer = og.Tokenizer(og_model)

        # Number of silence chunks for right-context flushing
        num_silence_chunks = 4

        def _transcribe_streaming(audio_arr: np.ndarray, genai_model) -> str:
            """Transcribe audio using stateful streaming processor."""
            audio = audio_arr.astype(np.float32)
            stream_processor = og.StreamingProcessor(genai_model)
            tokenizer_stream = tokenizer.create_stream()
            params = og.GeneratorParams(genai_model)
            generator = og.Generator(genai_model, params)

            transcript = ""

            def decode_tokens():
                nonlocal transcript
                while not generator.is_done():
                    generator.generate_next_token()
                    tokens = generator.get_next_tokens()
                    if len(tokens) > 0:
                        text = tokenizer_stream.decode(tokens[0])
                        if text:
                            transcript += text

            # Feed audio chunks
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start : start + chunk_samples].astype(np.float32)
                inputs = stream_processor.process(chunk)
                if inputs is not None:
                    generator.set_inputs(inputs)
                    decode_tokens()

            # Flush remaining audio in the processor
            inputs = stream_processor.flush()
            if inputs is not None:
                generator.set_inputs(inputs)
                decode_tokens()

            # Feed silence chunks for right-context flushing
            for _ in range(num_silence_chunks):
                silence = np.zeros(chunk_samples, dtype=np.float32)
                inputs = stream_processor.process(silence)
                if inputs is not None:
                    generator.set_inputs(inputs)
                    decode_tokens()

            return transcript

        all_preds = []
        all_targets = []
        total_audio_duration = 0.0
        total_inference_time = 0.0

        for batch in dataloader:
            input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)

            # Convert input to list of audio arrays
            audio_arrays = []
            if isinstance(input_data, (np.ndarray, torch.Tensor)):
                arr = np.array(input_data) if isinstance(input_data, torch.Tensor) else input_data
                if arr.ndim == 1:
                    audio_arrays = [arr]
                else:
                    audio_arrays = [arr[i] for i in range(arr.shape[0])]
            elif isinstance(input_data, list):
                audio_arrays = [np.array(a) if not isinstance(a, np.ndarray) else a for a in input_data]

            if not audio_arrays:
                continue

            start_time = time.perf_counter()
            for arr in audio_arrays:
                total_audio_duration += len(arr) / sample_rate
                transcription = _transcribe_streaming(arr, og_model)
                all_preds.append(transcription)
            total_inference_time += time.perf_counter() - start_time

            # Collect reference texts
            if isinstance(labels, (list, tuple)):
                all_targets.extend(labels)
            else:
                all_targets.append(labels)

        del og_model

        timing_metadata = {
            "total_audio_duration": total_audio_duration,
            "total_inference_time": total_inference_time,
        }
        return OliveModelOutput(preds=all_preds, logits=timing_metadata), all_targets

    def _evaluate_onnx_latency(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session, inference_settings = OnnxEvaluator.get_session_wrapper(
            model, metric, dataloader, device, execution_providers
        )
        io_config = model.io_config

        batch = next(iter(dataloader))
        input_data = OliveEvaluator.extract_input_data(batch)
        input_feed = format_data(input_data, io_config)

        latencies = session.time_run(
            input_feed,
            num_runs=repeat_test_num,
            num_warmup=warmup_num,
            sleep_time=sleep_num,
        )

        tuning_result_file = inference_settings.get("tuning_result_file")
        if tuning_result_file:
            dump_tuning_result(session.session, tuning_result_file)
        return latencies

    @staticmethod
    def _evaluate_distributed_accuracy_worker(config) -> tuple[list[Any], list[Any]]:
        model_path = config["model_path"]
        local_rank = config["local_rank"]
        world_size = config["world_size"]
        inference_settings = config["inference_settings"]
        execution_providers = config["providers"]
        metric = Metric.from_json(config["metric"])

        import os

        os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

        from mpi4py import MPI

        local_rank = MPI.COMM_WORLD.Get_rank()

        inference_settings["execution_provider"] = execution_providers
        inference_settings["provider_options"] = [
            {"device_id": str(local_rank)} if provider == "CUDAExecutionProvider" else {}
            for provider in execution_providers
        ]

        model = ONNXModelHandler(model_path, inference_settings=inference_settings)
        dataloader, _, post_func = OnnxEvaluator.get_user_config(model.framework, metric)

        session = model.prepare_session(inference_settings=inference_settings, device=Device.GPU, rank=int(local_rank))
        io_config = model.io_config

        preds = []
        targets = []
        logits = []
        output_names = io_config["output_names"]
        for batch in dataloader:
            input_data, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            input_dict = format_data(input_data, io_config)
            MPI.COMM_WORLD.barrier()  # Synchronize before starting each run
            output = session.run(None, input_dict)
            output = torch.Tensor(output[0]) if len(output_names) == 1 else torch.Tensor(output)
            post_output = post_func(output) if post_func else output
            preds.extend(post_output.tolist())
            targets.extend(labels.data.tolist())
            logits.extend(output.tolist())

        model_output = OliveModelOutput(preds=preds, logits=logits)
        return model_output, targets

    def _evaluate_distributed_accuracy(
        self,
        model: DistributedOnnxModelHandler,
        metric: Metric,
        device: Device,
        execution_providers: Union[str, list[str]],
    ) -> MetricResult:
        from mpi4py.futures import MPIPoolExecutor

        config = {
            "model_path": None,
            "local_rank": None,
            "world_size": model.num_ranks,
            "inference_settings": metric.get_inference_settings(Framework.ONNX.lower()),
            "metric": metric.to_json(),
        }

        args = []
        for rank in range(model.num_ranks):
            cfg = deepcopy(config)
            cfg["local_rank"] = rank
            cfg["model_path"] = model.ranked_model_path(rank)
            cfg["device"] = device
            cfg["providers"] = execution_providers
            args.append(cfg)

        with MPIPoolExecutor(max_workers=model.num_ranks) as executor:
            results = executor.map(OnnxEvaluator._evaluate_distributed_accuracy_worker, args)
            executor.shutdown()

        preds = [x for p, _, _ in results for x in p]
        targets = [x for _, t, _ in results for x in t]
        logits = [x for _, _, logit in results for x in logit]
        model_output = OliveModelOutput(preds, logits)
        return OliveEvaluator.compute_accuracy(metric, model_output, targets)

    @staticmethod
    def _evaluate_distributed_latency_worker(config) -> list[float]:
        model_path = config["model_path"]
        local_rank = config["local_rank"]
        world_size = config["world_size"]
        inference_settings = config["inference_settings"]
        execution_providers = config["providers"]
        metric = Metric.from_json(config["metric"])

        import os

        os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

        from mpi4py import MPI

        local_rank = MPI.COMM_WORLD.Get_rank()
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        inference_settings["execution_provider"] = execution_providers
        inference_settings["provider_options"] = [
            {"device_id": str(local_rank)} if provider == "CUDAExecutionProvider" else {}
            for provider in execution_providers
        ]

        model = ONNXModelHandler(model_path, inference_settings=inference_settings)
        dataloader, _, _ = OnnxEvaluator.get_user_config(model.framework, metric)
        session = model.prepare_session(inference_settings=inference_settings, device=Device.GPU, rank=int(local_rank))
        io_config = model.io_config

        batch = next(iter(dataloader))
        input_data = OliveEvaluator.extract_input_data(batch)
        input_feed = format_data(input_data, io_config)
        kv_cache_ortvalues = (
            {} if (metric.user_config and getattr(metric.user_config, "shared_kv_buffer", None)) else None
        )

        io_bind = OnnxEvaluator.io_bind_enabled(metric, model.inference_settings)
        if io_bind:
            io_bind_op = prepare_io_bindings(
                session,
                input_feed,
                Device.GPU,
                shared_kv_buffer=getattr(metric.user_config, "shared_kv_buffer", None) if metric.user_config else None,
                kv_cache_ortvalues=kv_cache_ortvalues,
            )
        latencies = []
        for i in range(warmup_num + repeat_test_num):
            MPI.COMM_WORLD.barrier()  # Synchronize before starting each run
            start_time = time.perf_counter()
            if io_bind:
                session.run_with_iobinding(io_bind_op)
            else:
                session.run(None, input_feed)
            if i > warmup_num:
                latencies.append(time.perf_counter() - start_time)
            time.sleep(sleep_num)

        return latencies

    def _evaluate_distributed_latency(
        self,
        model: DistributedOnnxModelHandler,
        metric: Metric,
        device,
        execution_providers: Union[str, list[str]],
    ) -> list[float]:
        from mpi4py.futures import MPIPoolExecutor

        config = {
            "model_path": None,
            "local_rank": None,
            "world_size": model.num_ranks,
            "inference_settings": metric.get_inference_settings(Framework.ONNX.lower()),
            "metric": metric.to_json(),
        }

        args = []
        for rank in range(model.num_ranks):
            cfg = deepcopy(config)
            cfg["local_rank"] = rank
            cfg["model_path"] = model.ranked_model_path(rank)
            cfg["device"] = device
            cfg["providers"] = execution_providers
            args.append(cfg)

        with MPIPoolExecutor(max_workers=model.num_ranks) as executor:
            results = executor.map(OnnxEvaluator._evaluate_distributed_latency_worker, args)
            executor.shutdown()

        return [x for r in results for x in r]

    def _evaluate_accuracy(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        if isinstance(model, ONNXModelHandler):
            return self._evaluate_onnx_accuracy(model, metric, dataloader, post_func, device, execution_providers)
        elif isinstance(model, DistributedOnnxModelHandler):
            if device != Device.GPU:
                raise ValueError("Distributed inferencing is supported only on GPU")
            return self._evaluate_distributed_accuracy(model, metric, device, execution_providers)
        else:
            raise TypeError(f"Cannot evaluate accuracy for model of type: {type(model)}")

    def _evaluate_raw_latency(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        if isinstance(model, ONNXModelHandler):
            return self._evaluate_onnx_latency(model, metric, dataloader, post_func, device, execution_providers)
        elif isinstance(model, DistributedOnnxModelHandler):
            if device != Device.GPU:
                raise ValueError("Distributed inferencing is supported only on GPU")
            return self._evaluate_distributed_latency(model, metric, device, execution_providers)
        else:
            raise TypeError(f"Cannot evaluate latency for model of type: {type(model)}")


@Registry.register(str(Framework.PYTORCH))
@Registry.register("PyTorchEvaluator")
class PyTorchEvaluator(_OliveEvaluator):
    @torch.no_grad()
    def _inference(
        self,
        model: "PyTorchModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        session = model.prepare_session()
        preds = []
        targets = []
        logits = []
        device = _OliveEvaluator.device_string_to_torch_device(device)
        run_kwargs = metric.get_run_kwargs()
        session.to(device)
        for batch in dataloader:
            input_data_i, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            input_data = tensor_data_to_device(input_data_i, device)
            result = model.run_session(session, input_data, **run_kwargs)
            outputs = post_func(result) if post_func else result
            # keep the outputs and results as torch tensor on cpu
            # it is expensive to convert to list and then convert back to torch tensor
            preds.append(outputs.cpu())
            targets.append(labels.cpu())
            try:
                if not isinstance(result, torch.Tensor) and getattr(result, "logits", None) is not None:
                    logits.append(result.logits.cpu())
                elif isinstance(result, tuple):
                    logits.append(result[0].cpu())
                else:
                    logits.append(result.cpu())
            except Exception as e:
                logger.warning("Error getting logits from PyTorch model output: %s", e)
                logits.append(torch.tensor([]))
        # concatenate along the batch dimension
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        logits = torch.cat(logits, dim=0)
        # move model to cpu
        if device:
            session.to("cpu")
        # only move to cpu cannot release gpu memory, call cuda.empty_cache() to release gpu memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return OliveModelOutput(preds=preds, logits=logits), targets

    def _evaluate_accuracy(
        self,
        model: "PyTorchModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        if _is_vision_metric(metric):
            _validate_vision_task_metric(metric)
            inference_output, targets = self._inference_vision(
                model, metric, dataloader, post_func, device, execution_providers
            )
        elif _is_text_based_metric(metric):
            inference_output, targets = self._inference_text(
                model, metric, dataloader, post_func, device, execution_providers
            )
        else:
            inference_output, targets = self._inference(
                model, metric, dataloader, post_func, device, execution_providers
            )
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    @torch.no_grad()
    def _inference_text(
        self,
        model: "PyTorchModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        """Text-based inference for speech/ASR metrics (WER, RTFx)."""
        session = model.prepare_session()
        all_preds = []
        all_targets = []
        total_audio_duration = 0.0
        total_inference_time = 0.0
        device = _OliveEvaluator.device_string_to_torch_device(device)
        run_kwargs = metric.get_run_kwargs()
        session.to(device)
        sample_rate = (
            metric.data_config.pre_process_data_config.params.get("sample_rate", 16000)
            if (metric.data_config and metric.data_config.pre_process_data_config)
            else 16000
        )

        for batch in dataloader:
            input_data_i, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            # Track audio duration from input data
            if isinstance(input_data_i, (np.ndarray, torch.Tensor)):
                audio_samples = input_data_i.shape[-1] if len(input_data_i.shape) > 1 else input_data_i.shape[0]
                total_audio_duration += audio_samples / sample_rate
            elif isinstance(input_data_i, dict):
                for v in input_data_i.values():
                    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 1:
                        total_audio_duration += v.shape[-1] / sample_rate
                        break

            input_data = tensor_data_to_device(input_data_i, device)
            start_time = time.perf_counter()
            result = model.run_session(session, input_data, **run_kwargs)
            outputs = post_func(result) if post_func else result
            total_inference_time += time.perf_counter() - start_time

            if isinstance(outputs, str):
                all_preds.append(outputs)
            elif isinstance(outputs, (list, tuple)):
                if not outputs:
                    continue
                if not isinstance(outputs[0], str):
                    raise ValueError(
                        f"post_func must return str or list[str] for text-based metrics (WER), "
                        f"but got list of {type(outputs[0]).__name__}. "
                        f"Ensure your post_func decodes model output to text."
                    )
                all_preds.extend(outputs)
            else:
                raise ValueError(
                    f"post_func must return str or list[str] for text-based metrics (WER), "
                    f"but got {type(outputs).__name__}. "
                    f"Ensure your post_func decodes model output to text."
                )
            if isinstance(labels, (list, tuple)):
                all_targets.extend(labels)
            else:
                all_targets.append(labels)
        if device:
            session.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        timing_metadata = {
            "total_audio_duration": total_audio_duration,
            "total_inference_time": total_inference_time,
        }
        return OliveModelOutput(preds=all_preds, logits=timing_metadata), all_targets

    @torch.no_grad()
    def _inference_vision(
        self,
        model: "PyTorchModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        """Vision-based inference for VQA/OCR metrics (exact_match, relaxed_accuracy, word_sort_ratio)."""
        session = model.prepare_session()
        all_preds = []
        all_targets = []
        torch_device = _OliveEvaluator.device_string_to_torch_device(device)
        run_kwargs = metric.get_run_kwargs()
        session.to(torch_device)

        for batch in dataloader:
            input_data_i, labels = OliveEvaluator.unpack_batch_for_accuracy(batch)
            input_data = tensor_data_to_device(input_data_i, torch_device)
            result = model.run_session(session, input_data, **run_kwargs)
            outputs = post_func(result) if post_func else result

            if isinstance(outputs, str):
                all_preds.append(outputs)
            elif isinstance(outputs, (list, tuple)):
                if not outputs:
                    continue
                if not isinstance(outputs[0], str):
                    raise ValueError(
                        f"post_func must return str or list[str] for vision metrics, "
                        f"but got list of {type(outputs[0]).__name__}. "
                        f"Ensure your post_func decodes model output to answer text."
                    )
                all_preds.extend(outputs)
            else:
                raise ValueError(
                    f"post_func must return str or list[str] for vision metrics, "
                    f"but got {type(outputs).__name__}. "
                    f"Ensure your post_func decodes model output to answer text."
                )
            if isinstance(labels, (list, tuple)):
                all_targets.extend(labels)
            else:
                all_targets.append(labels)

        if torch_device:
            session.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return OliveModelOutput(preds=all_preds, logits=None), all_targets

    @torch.no_grad()
    def _evaluate_raw_latency(
        self,
        model: "PyTorchModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        # pylint: disable=expression-not-assigned
        warmup_num, repeat_test_num, _ = get_latency_config_from_metric(metric)
        # pytorch model doesn't use inference_settings, so we can pass None
        session = model.prepare_session(inference_settings=None, device=device)

        batch = next(iter(dataloader))
        input_data = OliveEvaluator.extract_input_data(batch)
        torch_device = _OliveEvaluator.device_string_to_torch_device(device)
        run_kwargs = metric.get_run_kwargs()

        session.to(torch_device)
        input_data = tensor_data_to_device(input_data, torch_device)

        # warm up
        for _ in range(warmup_num):
            model.run_session(session, input_data, **run_kwargs)

        latencies = []
        if torch_device == torch.device("cuda"):
            # synchronize before starting the test
            torch.cuda.synchronize()
            # cuda events for measuring latency
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(repeat_test_num):
                starter.record()
                model.run_session(session, input_data, **run_kwargs)
                ender.record()
                # synchronize after forward pass
                torch.cuda.synchronize()
                # add time in seconds, originally in milliseconds
                latencies.append(starter.elapsed_time(ender) * 1e-3)

            # move model back to cpu
            session.to("cpu")
            tensor_data_to_device(input_data, Device.CPU)

            # only move to cpu cannot release gpu memory, call cuda.empty_cache() to release gpu memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            for _ in range(repeat_test_num):
                t = time.perf_counter()
                # TODO(jambayk): do we care about the efficiency of if/else here?
                # probably won't add much overhead compared to the inference time
                # also we are doing the same for all models
                model.run_session(session, input_data, **run_kwargs)
                latencies.append(time.perf_counter() - t)

        return latencies


@Registry.register(str(Framework.OPENVINO))
@Registry.register("OpenVINOEvaluator")
class OpenVINOEvaluator(_OliveEvaluator):
    def _inference(
        self,
        model: "OpenVINOModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.OPENVINO.lower()), device=device
        )
        run_kwargs = metric.get_run_kwargs()

        preds = []
        targets = []
        logits = []
        for batch in dataloader:
            input_data, label = OliveEvaluator.unpack_batch_for_accuracy(batch)
            model.run_session(session, {0: input_data}, **run_kwargs)
            result = session.get_output_tensor(0).data
            outputs = post_func(result) if post_func else result
            preds.extend(outputs)
            targets.extend(label)
            logits.extend(result)
        return OliveModelOutput(preds=preds, logits=logits), targets

    def _evaluate_accuracy(
        self,
        model: "OpenVINOModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        inference_output, targets = self._inference(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    def _evaluate_raw_latency(
        self,
        model: "OpenVINOModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.OPENVINO.lower()), device=device
        )
        run_kwargs = metric.get_run_kwargs()

        latencies = []
        for batch in dataloader:
            input_data = OliveEvaluator.extract_input_data(batch)
            t = time.perf_counter()
            model.run_session(session, input_data, **run_kwargs)
            latencies.append(time.perf_counter() - t)
        return latencies


@Registry.register(str(Framework.QNN))
@Registry.register("QNNEvaluator")
class QNNEvaluator(_OliveEvaluator):
    def _inference(
        self,
        model: "QNNModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> tuple[OliveModelOutput, Any]:
        dataloader = self._prepare_dataloader(dataloader, model)
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.QNN.lower()), device=device
        )

        preds = []
        targets = []
        logits = []
        run_kwargs = metric.get_run_kwargs()
        for data_dir, input_list, labels in dataloader:
            if labels is None:
                raise ValueError("Accuracy evaluation requires dataset with labels.")
            run_kwargs["data_dir"] = data_dir
            result = model.run_session(session, input_list, **run_kwargs)
            for idx, output in enumerate(result.get("result")):
                if post_func:
                    post_output = post_func(output)
                else:
                    raise ValueError("Post processing function is required for QNN model")
                preds.extend(post_output.tolist())
                if isinstance(labels[idx], (list, np.ndarray)):
                    targets.extend(labels[idx])
                else:
                    targets.append(labels[idx])
                logits.extend(output.tolist())
        return OliveModelOutput(preds=preds, logits=logits), targets

    def _evaluate_accuracy(
        self,
        model: "QNNModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        inference_output, targets = self._inference(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    def _evaluate_raw_latency(
        self,
        model: "QNNModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> list[float]:
        dataloader = self._prepare_dataloader(dataloader, model, 1)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.QNN.lower()), device=device
        )

        data_dir, input_data, _ = next(iter(dataloader))
        # for qnn-net-run only keep 20 logs
        total_runs = min(warmup_num + repeat_test_num, 20)
        run_kwargs = metric.get_run_kwargs()
        run_kwargs["data_dir"] = data_dir
        run_kwargs["runs"] = total_runs
        run_kwargs["sleep"] = sleep_num
        results = model.run_session(session, input_data, **run_kwargs)
        return results["latencies"]["net_run"][warmup_num:]

    def _prepare_dataloader(
        self, dataloader: "DataLoader", model: "QNNModelHandler", file_chunk_size=None
    ) -> FileListDataLoader:
        if isinstance(dataloader, FileListDataLoader):
            return dataloader
        return FileListCommonDataLoader(dataloader, model.io_config, batch_size=file_chunk_size)


@Registry.register("LMEvaluator")
class LMEvaluator(OliveEvaluator):
    def __init__(self, tasks: list[str], **kwargs):
        super().__init__(**kwargs)

        self.tasks = tasks
        self.limit = kwargs.get("limit")
        self.model_class = kwargs.get("model_class")
        self.batch_size = kwargs.get("batch_size", 1)
        self.max_length = kwargs.get("max_length")
        # Preserve lm-eval bootstrap control from recipe configs. Some generation metrics disable
        # bootstrap stderr resampling to avoid extra post-processing work on large test sets.
        self.bootstrap_iters = kwargs.get("bootstrap_iters", 100000)
        self.ep = kwargs.get("execution_provider")
        self.ep_options = kwargs.get("provider_options")
        self.device = kwargs.get("device")

    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: list[Metric],
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        from lm_eval import simple_evaluate
        from lm_eval.api.registry import get_model
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import setup_logging

        import olive.evaluator.lmeval_ort  # noqa: F401 # pylint: disable=unused-import

        setup_logging("ERROR")

        if not self.model_class:
            if isinstance(model, HfModelHandler):
                self.model_class = "hf"
            elif isinstance(model, ONNXModelHandler):
                self.model_class = "ort"
            else:
                raise ValueError("Failed to automatically deduce model class. Provide it in user input!")

        init_args = {}
        device = _OliveEvaluator.device_string_to_torch_device(self.device or device)
        if self.model_class == "hf":
            init_args = {
                "pretrained": model.load_model(cache_model=False).eval().to(device),
                "tokenizer": model.get_hf_tokenizer(),
                "device": device,
            }
        elif self.model_class == "ort":
            init_args = {
                "model_path": model.model_path,
                "ep": self.ep or execution_providers,
                "ep_options": self.ep_options,
            }
        elif self.model_class == "ortgenai":
            init_args = {
                "pretrained": _get_genai_model_dir(model),
                "ep": self.ep or execution_providers,
                "ep_options": self.ep_options,
                "device": device,
            }
        else:
            raise ValueError(f"Unknown model class: {self.model_class}")

        logger.debug(
            "Running LM evaluation with model class: %s and device/ep args: %s",
            self.model_class,
            {k: v for k, v in init_args.items() if k in ["device", "ep", "ep_options"]},
        )

        metrics = {}
        if MetricType.SIZE_ON_DISK.value in self.tasks:
            self.tasks.remove(MetricType.SIZE_ON_DISK.value)
            metrics[MetricType.SIZE_ON_DISK.value] = MetricResult.model_validate(
                {
                    SizeOnDiskSubType.BYTES.value: {
                        "value": model.size_on_disk,
                        "priority": -1,
                        "higher_is_better": False,
                    }
                }
            )

        if self.tasks:
            lmmodel = get_model(self.model_class)(**init_args, batch_size=self.batch_size, max_length=self.max_length)

            results = simple_evaluate(
                model=lmmodel,
                tasks=self.tasks,
                task_manager=TaskManager(),
                log_samples=False,
                batch_size=self.batch_size,
                device=device,
                limit=self.limit,
                # Forward the configured value instead of letting lm-eval silently use its default.
                bootstrap_iters=self.bootstrap_iters,
            )

            for task_name in sorted(results["results"].keys()):
                metric_items = sorted(results["results"][task_name].items())

                task_metrics = {}
                for mf, v in metric_items:
                    if mf == "alias":
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    if "," in mf:
                        m, _ = mf.split(",", 1)
                    else:
                        m = mf
                    if not m.endswith("_stderr"):
                        task_metrics[m] = SubMetricResult(value=v, priority=-1, higher_is_better=True)

                metrics[task_name] = MetricResult.model_validate(task_metrics)

        return flatten_metric_result(metrics)


@Registry.register("MTEBEvaluator")
class MTEBEvaluator(OliveEvaluator):
    """Evaluator for embedding models using the MTEB (Massive Text Embedding Benchmark) library.

    Supports three model classes, mirroring :class:`LMEvaluator`:

    - ``"hf"`` — evaluates a HuggingFace model via sentence-transformers
    - ``"ort"`` — evaluates a plain ONNX model via ORT inference session
    - ``"ortgenai"`` — evaluates an ORT-GenAI model (ModelBuilder output)

    Example recipe config::

        "evaluators": {
            "evaluator": {
                "type": "MTEBEvaluator",
                "tasks": ["STS17"],
                "batch_size": 32
            }
        },
        "evaluator": "evaluator"
    """

    def __init__(self, tasks: list[str], **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.batch_size = kwargs.get("batch_size", 32)
        self.max_length = kwargs.get("max_length")
        self.model_class = kwargs.get("model_class")
        self.ep = kwargs.get("execution_provider")
        self.ep_options = kwargs.get("provider_options")
        self.eval_splits = kwargs.get("eval_splits")
        self.eval_subsets = kwargs.get("eval_subsets")
        self.output_folder = kwargs.get("output_folder")

    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: list[Metric],
        device: Device = Device.CPU,
        execution_providers: Optional[Union[str, list[str]]] = None,
    ) -> MetricResult:
        import mteb

        from olive.evaluator.mteb_ort import MTEBORTEvaluator, MTEBORTGenAIEvaluator

        # Auto-detect model class from the model handler
        model_class = self.model_class
        if not model_class:
            if isinstance(model, HfModelHandler):
                model_class = "hf"
            elif isinstance(model, ONNXModelHandler):
                # ModelBuilder outputs ONNXModelHandler but with genai_config.json
                genai_config_path = _find_genai_config(model)
                model_class = "ortgenai" if genai_config_path is not None else "ort"
            else:
                raise ValueError(
                    "Unable to auto-detect model_class for MTEBEvaluator from model handler "
                    f"{type(model).__name__}. Please set model_class explicitly to one of "
                    "'hf', 'ort', or 'ortgenai'."
                )

        logger.info("Running MTEB evaluation with model_class=%s, tasks=%s", model_class, self.tasks)

        # Build the MTEB-compatible model wrapper
        if model_class == "hf":
            from sentence_transformers import SentenceTransformer

            # Map Olive Device to PyTorch device string (Olive uses "gpu", PyTorch expects "cuda")
            device_str = device.value if isinstance(device, Device) else str(device)
            normalized = device_str.lower()
            if normalized == "gpu":
                sentence_transformer_device = "cuda"
            elif normalized.startswith("gpu:"):
                sentence_transformer_device = f"cuda{device_str[3:]}"
            else:
                sentence_transformer_device = device_str
            mteb_model = SentenceTransformer(model.model_name_or_path, device=sentence_transformer_device)
        elif model_class == "ort":
            mteb_model = MTEBORTEvaluator(
                model_path=model.model_path,
                batch_size=self.batch_size,
                max_length=self.max_length,
                ep=self.ep
                or (execution_providers[0] if isinstance(execution_providers, list) else execution_providers),
                ep_options=self.ep_options,
            )
        elif model_class == "ortgenai":
            mteb_model = MTEBORTGenAIEvaluator(
                pretrained=_get_genai_model_dir(model),
                batch_size=self.batch_size,
                max_length=self.max_length,
                ep=self.ep
                or (execution_providers[0] if isinstance(execution_providers, list) else execution_providers)
                or "follow_config",
                ep_options=self.ep_options,
            )
        else:
            raise ValueError(f"Unknown model class for MTEBEvaluator: {model_class}")

        # Run MTEB evaluation
        mteb_tasks = mteb.get_tasks(tasks=self.tasks)
        evaluation = mteb.MTEB(tasks=mteb_tasks)

        run_kwargs = {}
        if self.eval_splits:
            run_kwargs["eval_splits"] = self.eval_splits
        if self.eval_subsets:
            run_kwargs["eval_subsets"] = self.eval_subsets

        task_results = evaluation.run(
            mteb_model,
            output_folder=self.output_folder,
            overwrite_results=True,
            verbosity=0,
            **run_kwargs,
        )

        # Convert MTEB results into Olive MetricResult
        metrics_dict = {}
        for task_result in task_results:
            task_name = task_result.task_name
            task_metrics = {
                "main_score": SubMetricResult(value=task_result.main_score, priority=-1, higher_is_better=True),
            }
            for split_name, split_scores in task_result.scores.items():
                for lang_score in split_scores:
                    subset = lang_score.get("hf_subset", "")
                    score_key = f"{split_name}_{subset}" if subset else split_name
                    task_metrics[score_key] = SubMetricResult(
                        value=lang_score.get("main_score", 0.0),
                        priority=-1,
                        higher_is_better=True,
                    )
            metrics_dict[task_name] = MetricResult.model_validate(task_metrics)

        return flatten_metric_result(metrics_dict)


class OliveEvaluatorConfig(NestedConfig):
    _nested_field_name: ClassVar[str] = "type_args"

    name: Optional[str] = None
    type: Optional[str] = None
    type_args: dict = Field(default_factory=dict)

    # user script to define and register the evaluator
    user_script: Optional[Union[Path, str]] = None
    script_dir: Optional[Union[Path, str]] = None

    metrics: list[Metric] = []  # noqa: RUF012

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # call import_user_module to load the user script once and register the evaluator
        if self.user_script:
            import_user_module(self.user_script, self.script_dir)

    @property
    def is_accuracy_drop_tolerant(self):
        for metric in self.metrics:
            for sub_metric in metric.sub_types:
                if metric.type == MetricType.ACCURACY and sub_metric.higher_is_better:
                    return sub_metric.goal is not None and sub_metric.goal.has_regression_goal()
        return False

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values):
        # In pydantic v2, values can be None when no arguments are provided
        if values is None:
            values = {}

        if values.get("user_script"):
            import_user_module(values["user_script"], values.get("script_dir"))

        evaluator_type = values.get("type")
        if evaluator_type is not None and Registry.get(evaluator_type) is None:
            raise ValueError(f"Invalid/unknown evaluator type: {evaluator_type}")

        return values

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
        metric_len = len(v)

        metric_names = {metric.name for metric in v}
        assert len(metric_names) == metric_len, "Metric names must be unique"

        sub_type_names = set()
        sub_type_with_rank = set()
        rank_set = set()
        for metric in v:
            for sub_type in metric.sub_types:
                unique_metric_name = joint_metric_key(metric.name, sub_type.name)
                sub_type_names.add(unique_metric_name)
                if sub_type.priority != -1:
                    sub_type_with_rank.add(unique_metric_name)
                    rank_set.add(sub_type.priority)

        if not rank_set and len(sub_type_names) == 1:
            logger.debug(
                "No priority is specified, but only one sub type "
                " metric is specified. Use rank 1 for single for this metric."
            )
            v[0].sub_types[0].priority = 1
        elif not rank_set and len(sub_type_names) > 1:
            raise ValueError("Priority must be specified for multiple sub type metrics")

        expected_rank_set = set(range(1, len(sub_type_with_rank) + 1))
        # Check if all ranks are present
        if rank_set != expected_rank_set:
            raise ValueError(
                f"Priorities must be unique and in the range 1 to {metric_len}\n"
                f"\tActual: {rank_set}\n\tExpected: {expected_rank_set}"
            )

        return v

    def create_evaluator(self, model: "OliveModelHandler" = None) -> OliveEvaluator:
        return Registry.get(self.type or str(model.framework))(**(self.type_args or {}))
