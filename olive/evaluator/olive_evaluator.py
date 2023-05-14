# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from numbers import Number

import numpy as np
import torch
from pydantic import validator
from torch.utils.data import Dataset

from olive.common.config_utils import ConfigBase
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import tensor_data_to_device
from olive.constants import Framework
from olive.evaluator.accuracy import AUC, AccuracyScore, F1Score, Precision, Recall
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.evaluator.metric_config import joint_metric_key, MetricResult, SubTypeMetricResult
from olive.model import OliveModel, ONNXModel, OpenVINOModel, PyTorchModel, SNPEModel
from olive.systems.common import Device

logger = logging.getLogger(__name__)


class DummyDataloader(Dataset):
    def __init__(self, input_names, input_shapes, input_types):
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.input_types = input_types

    def __len__(self):
        return 100

    def __getitem__(self, index):
        str_to_type = {"float32": torch.float32, "float16": torch.float16, "int32": torch.int32, "int64": torch.int64}
        input_types = []
        if self.input_types:
            for input_type in self.input_types:
                input_types.append(str_to_type[input_type])
        else:
            for _ in range(len(self.input_names)):
                input_types.append(torch.float32)
        if len(self.input_names) == 1:
            dummy_inputs = torch.ones(self.input_shapes[0], dtype=input_types[0])
        else:
            dummy_inputs = {}
            for input_name, input_shape, input_type in zip(self.input_names, self.input_shapes, input_types):
                dummy_inputs.update({input_name: torch.ones(input_shape, dtype=input_type)})
        label = 0
        return dummy_inputs, label


class OliveEvaluator(ABC):
    registry: Dict[str, "OliveEvaluator"] = {}

    @classmethod
    def __init_subclass__(cls, framework: Framework, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.framework = framework
        cls.registry[str(framework).lower()] = cls

    def __init__(self):
        pass

    def get_inference_settings(self, metric: Metric) -> Dict[str, Any]:
        # user.config.inference_settings > model.inference_settings > default inference_settings
        # when user.config.inference_settings is None, the model.inference_settings
        # will be used in model.prepare_session(..)
        return (
            metric.user_config.inference_settings.get(self.framework.lower())
            if metric.user_config.inference_settings
            else None
        )

    @abstractmethod
    def _evaluate_accuracy(
        self, model: OliveModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        raise NotImplementedError()

    @abstractmethod
    def _evaluate_latency(
        self, model: OliveModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        raise NotImplementedError()

    def _evaluate_custom(
        self,
        model: OliveModel,
        metric: Metric,
        dataloader: Dataset,
        eval_func,
        device: Device = Device.CPU,
        post_func=None,
    ) -> MetricResult:
        # TODO: Change the evaluate function to accept the metric rather than
        # breaking it into multiple arguments
        # return eval_func(model, metric, dataloader, device, post_func)
        raw_res = eval_func(model, metric.user_config.data_dir, metric.user_config.batch_size, device)
        metric_res = {}
        for sub_type in metric.sub_types:
            if isinstance(raw_res, Number):
                assert len(metric.sub_types) == 1, "Only one sub type is allowed for single value custom metric"
                metric_res[sub_type.name] = SubTypeMetricResult(
                    value=raw_res, priority_rank=sub_type.priority_rank, higher_is_better=sub_type.higher_is_better
                )
            elif isinstance(raw_res, Dict):
                assert sub_type.name in raw_res, f"Custom metric {sub_type.name} is not in the result"
                metric_res[sub_type.name] = SubTypeMetricResult(
                    value=raw_res[sub_type.name],
                    priority_rank=sub_type.priority_rank,
                    higher_is_better=sub_type.higher_is_better,
                )
        return MetricResult.parse_obj(metric_res)

    def evaluate(self, model: OliveModel, metrics: List[Metric], device: Device = Device.CPU) -> MetricResult:
        metrics_res = {}
        for metric in metrics:
            dataloader, eval_func, post_func = OliveEvaluator.get_user_config(metric)

            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = self._evaluate_accuracy(model, metric, dataloader, device, post_func)
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = self._evaluate_latency(model, metric, dataloader, device, post_func)
            elif metric.type == MetricType.CUSTOM:
                metrics_res[metric.name] = self._evaluate_custom(
                    model, metric, dataloader, eval_func, device, post_func
                )
            else:
                raise TypeError(f"{metric.type} is not a supported metric type")
        return metrics_res

    @staticmethod
    def get_user_config(metric: Metric):
        user_module = UserModuleLoader(metric.user_config.user_script, metric.user_config.script_dir)

        post_processing_func = getattr(metric.user_config, "post_processing_func", None)
        post_func = user_module.load_object(post_processing_func)

        dataloader_func = getattr(metric.user_config, "dataloader_func", None)
        dataloader = user_module.call_object(
            dataloader_func, metric.user_config.data_dir, metric.user_config.batch_size
        )

        evaluate_func = getattr(metric.user_config, "evaluate_func", None)
        eval_func = user_module.load_object(evaluate_func)

        if metric.user_config.input_names and metric.user_config.input_shapes and not dataloader and not eval_func:
            dataloader = DummyDataloader(
                metric.user_config.input_names, metric.user_config.input_shapes, metric.user_config.input_types
            )

        if not dataloader or not post_func:
            dc = metric.data_config.to_data_container()

            # TODO remove user_scripts dataloader: we should respect user scripts
            # dataloder to meet back compatibility for time being.
            dataloader = dataloader or dc.create_dataloader()
            post_func = post_func or dc.config.post_process

        return dataloader, eval_func, post_func

    @staticmethod
    def compute_accuracy(metric: Metric, preds: Any, targets: Any) -> MetricResult:
        """
        Compute accuracy metrics
        """
        metric_res = {}
        sub_type_metric_value = None
        sub_types = metric.sub_types
        for sub_type in sub_types:
            metric_config = sub_type.metric_config
            if sub_type.name == AccuracySubType.ACCURACY_SCORE:
                sub_type_metric_value = AccuracyScore(metric_config).measure(preds, targets)
            elif sub_type.name == AccuracySubType.F1_SCORE:
                sub_type_metric_value = F1Score(metric_config).measure(preds, targets)
            elif sub_type.name == AccuracySubType.PRECISION:
                sub_type_metric_value = Precision(metric_config).measure(preds, targets)
            elif sub_type.name == AccuracySubType.RECALL:
                sub_type_metric_value = Recall(metric_config).measure(preds, targets)
            elif sub_type.name == AccuracySubType.AUC:
                sub_type_metric_value = AUC(metric_config).measure(preds, targets)
            else:
                raise TypeError(f"{sub_type} is not a accuracy metric supported")
            metric_res[sub_type.name] = SubTypeMetricResult(
                value=sub_type_metric_value,
                priority_rank=sub_type.priority_rank,
                higher_is_better=sub_type.higher_is_better,
            )
        return MetricResult.parse_obj(metric_res)

    @staticmethod
    def compute_latency(metric: Metric, latencies: Any) -> MetricResult:
        """
        Compute latency metrics
        """
        latency_metrics = {
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
        metric_res = {}
        for sub_type in metric.sub_types:
            metric_res[sub_type.name] = SubTypeMetricResult(
                value=latency_metrics[sub_type.name],
                priority_rank=sub_type.priority_rank,
                higher_is_better=sub_type.higher_is_better,
            )
        return MetricResult.parse_obj(metric_res)


class OnnxEvaluator(OliveEvaluator, framework=Framework.ONNX):
    def __init__(self):
        super().__init__()

    @staticmethod
    def format_input(input_data, io_config):
        """
        Format input data to ONNX input format.
        """
        input_names = io_config["input_names"]
        name_to_type = {k: v for k, v in zip(io_config["input_names"], io_config["input_types"])}
        if not isinstance(input_data, dict):
            input_data = dict(zip(input_names, [input_data]))
        input_dict = {
            k: np.ascontiguousarray(
                input_data[k].cpu().numpy() if isinstance(input_data[k], torch.Tensor) else input_data[k],
                dtype=name_to_type[k],
            )
            for k in input_data.keys()
            if k in input_names
        }
        return input_dict

    def _evaluate_accuracy(
        self, model: ONNXModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> Dict[str, Any]:
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)
        io_config = model.get_io_config()

        preds = []
        targets = []
        output_names = io_config["output_names"]
        for input_data, labels in dataloader:
            input_dict = OnnxEvaluator.format_input(input_data, io_config)
            res = session.run(input_feed=input_dict, output_names=None)
            result = torch.Tensor(res[0]) if len(output_names) == 1 else torch.Tensor(res)
            outputs = post_func(result) if post_func else result
            preds.extend(outputs.tolist())
            targets.extend(labels.data.tolist())

        return OliveEvaluator.compute_accuracy(metric, preds, targets)

    def _evaluate_latency(
        self, model: OliveModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> Dict[str, Any]:
        warmup_num, repeat_test_num, sleep_num = None, None, None
        for sub_type in metric.sub_types:
            if sub_type.metric_config:
                warmup_num = sub_type.metric_config.warmup_num
                repeat_test_num = sub_type.metric_config.repeat_test_num
                sleep_num = sub_type.metric_config.sleep_num
                break

        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)
        io_config = model.get_io_config()

        input_data, _ = next(iter(dataloader))
        input_dict = OnnxEvaluator.format_input(input_data, io_config)

        if metric.user_config.io_bind:
            io_bind_op = session.io_binding()
            io_bind_device = "cuda" if device == "gpu" else "cpu"
            for k, v in input_dict.items():
                io_bind_op.bind_cpu_input(k, v)
            for item in session.get_outputs():
                io_bind_op.bind_output(item.name, io_bind_device)

        for _ in range(warmup_num):
            if metric.user_config.io_bind:
                session.run_with_iobinding(io_bind_op)
            else:
                session.run(input_feed=input_dict, output_names=None)

        latencies = []
        for _ in range(repeat_test_num):
            if metric.user_config.io_bind:
                t = time.perf_counter()
                session.run_with_iobinding(io_bind_op)
                latencies.append(time.perf_counter() - t)
            else:
                t = time.perf_counter()
                session.run(input_feed=input_dict, output_names=None)
                latencies.append(time.perf_counter() - t)
            time.sleep(sleep_num)

        return OliveEvaluator.compute_latency(metric, latencies)


class PyTorchEvaluator(OliveEvaluator, framework=Framework.PYTORCH):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _device_string_to_torch_device(device: Device):
        return torch.device("cuda") if device == Device.GPU else torch.device(device)

    def _evaluate_accuracy(
        self, model: PyTorchModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        preds = []
        targets = []
        device = PyTorchEvaluator._device_string_to_torch_device(device)
        if device:
            session.to(device)
        for input_data, labels in dataloader:
            input_data = tensor_data_to_device(input_data, device)
            result = session(**input_data) if isinstance(input_data, dict) else session(input_data)
            outputs = post_func(result) if post_func else result
            # use the list.extend instead of list.append to avoid the different sub-array has different size when
            # batch size is greater than 2 so that the residue array has different size with the batch size,
            # which will result the exception like:
            #  ValueError: expected sequence of length 128 at dim 1 (got 3)
            preds.extend(outputs.tolist())
            targets.extend(labels.data.tolist())

        return OliveEvaluator.compute_accuracy(metric, preds, targets)

    def _evaluate_latency(
        self, model: PyTorchModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        warmup_num, repeat_test_num = None, None
        for sub_type in metric.sub_types:
            if sub_type.metric_config:
                warmup_num = sub_type.metric_config.warmup_num
                repeat_test_num = sub_type.metric_config.repeat_test_num
                break

        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        input_data, _ = next(iter(dataloader))
        device = PyTorchEvaluator._device_string_to_torch_device(device)
        if device:
            session.to(device)
            input_data = tensor_data_to_device(input_data, device)

        latencies = []
        if isinstance(input_data, dict):
            for _ in range(warmup_num):
                session(**input_data)
            for _ in range(repeat_test_num):
                t = time.perf_counter()
                session(**input_data)
                latencies.append(time.perf_counter() - t)
        else:
            for _ in range(warmup_num):
                session(input_data)
            for _ in range(repeat_test_num):
                t = time.perf_counter()
                session(input_data)
                latencies.append(time.perf_counter() - t)

        return OliveEvaluator.compute_latency(metric, latencies)


class SNPEEvaluator(OliveEvaluator, framework=Framework.SNPE):
    def __init__(self):
        super().__init__()

    def _evaluate_accuracy(
        self, model: SNPEModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        preds = []
        targets = []
        for data_dir, input_list, labels in dataloader:
            result = session(input_list, data_dir)
            if post_func:
                outputs = post_func(result)
            else:
                raise ValueError("Post processing function is required for SNPE model")
            preds.extend(outputs.tolist())
            targets.extend(labels.tolist())

        return OliveEvaluator.compute_accuracy(metric, preds, targets)

    def _evaluate_latency(
        self, model: SNPEModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        warmup_num, repeat_test_num, sleep_num = None, None, None
        for sub_type in metric.sub_types:
            if sub_type.metric_config:
                warmup_num = sub_type.metric_config.warmup_num
                repeat_test_num = sub_type.metric_config.repeat_test_num
                sleep_num = sub_type.metric_config.sleep_num
                break
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        data_dir, input_data, _ = next(iter(dataloader))
        total_runs = warmup_num + repeat_test_num
        results = session(input_data, data_dir, runs=total_runs, sleep=sleep_num)
        latencies = results["latencies"]["total_inference_time"][warmup_num]

        return OliveEvaluator.compute_latency(metric, latencies)


class OpenVINOEvaluator(OliveEvaluator, framework=Framework.OPENVINO):
    def __init__(self):
        super().__init__()

    def _evaluate_accuracy(
        self, model: OpenVINOModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        preds = []
        targets = []
        for input_data, labels in dataloader:
            result = session.infer_new_request({0: input_data})
            outputs = post_func(result) if post_func else result
            if not isinstance(labels, list):
                labels = [labels]
            preds.extend(outputs)
            targets.extend(labels)

        return OliveEvaluator.compute_accuracy(metric, preds, targets)

    def _evaluate_latency(
        self, model: OpenVINOModel, metric: Metric, dataloader: Dataset, device: Device = Device.CPU, post_func=None
    ) -> MetricResult:
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        latencies = []
        for input_data, _ in dataloader:
            t = time.perf_counter()
            session(input_data)
            latencies.append(time.perf_counter() - t)

        return OliveEvaluator.compute_latency(metric, latencies)


class OliveEvaluatorFactory:
    @staticmethod
    def create_evaluator_for_model(model: OliveModel) -> OliveEvaluator:
        evaluator_cls = OliveEvaluator.registry[str(model.framework).lower()]
        return evaluator_cls()


class OliveEvaluatorConfig(ConfigBase):
    metrics: List[Metric] = []

    @validator("metrics")
    def validate_metrics(cls, v):
        metric_len = len(v)
        if metric_len == 1:
            return v

        metric_names = set([metric.name for metric in v])
        assert len(metric_names) == metric_len, "Metric names must be unique"

        sub_type_names = set()
        sub_type_with_rank = set()
        rank_set = set()
        for metric in v:
            for sub_type in metric.sub_types:
                sub_type_names.add(joint_metric_key(metric.name, sub_type.name))
                if sub_type.priority_rank != -1:
                    sub_type_with_rank.add(sub_type.name)
                    rank_set.add(sub_type.priority_rank)

        if not rank_set and len(sub_type_names) == 1:
            logger.debug(
                """No priority rank is specified, but only one sub type
                metric is specified. Use rank 1 for single for this metric."""
            )
            v[0].sub_types[0].priority_rank = 1
        elif not rank_set and len(sub_type_names) > 1:
            raise ValueError("Priority rank must be specified for multiple sub type metrics")

        expected_rank_set = set(range(1, len(sub_type_with_rank) + 1))
        # Check if all ranks are present
        if rank_set != expected_rank_set:
            raise ValueError(f"Priority ranks must be unique and in the range 1 to {metric_len}")

        return v
