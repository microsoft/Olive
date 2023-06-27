# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from pydantic import validator
from torch.utils.data import Dataset

import olive.data.template as data_config_template
from olive.cache import get_local_path
from olive.common.config_utils import ConfigBase
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import tensor_data_to_device
from olive.constants import Framework
from olive.evaluator.metric import (
    LatencySubType,
    Metric,
    MetricResult,
    MetricType,
    SubMetricResult,
    flatten_metric_result,
    get_latency_config_from_metric,
    joint_metric_key,
)
from olive.evaluator.metric_backend import MetricBackend
from olive.hardware import Device
from olive.model import DistributedOnnxModel, OliveModel, ONNXModel, OpenVINOModel, PyTorchModel, SNPEModel
from olive.model.model_config import is_io_config_static
from olive.snpe.data_loader import SNPECommonDataLoader, SNPEDataLoader

logger = logging.getLogger(__name__)


class OliveEvaluator(ABC):
    registry: Dict[str, Type["OliveEvaluator"]] = {}

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
        self,
        model: OliveModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        raise NotImplementedError()

    @abstractmethod
    def _evaluate_latency(
        self,
        model: OliveModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        raise NotImplementedError()

    def _evaluate_custom(
        self,
        model: OliveModel,
        metric: Metric,
        dataloader: Dataset,
        eval_func,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers=None,
    ) -> MetricResult:
        # TODO: Change the evaluate function to accept the metric rather than
        # breaking it into multiple arguments
        # return eval_func(model, metric, dataloader, device, post_func)
        raw_res = eval_func(
            model,
            get_local_path(metric.user_config.data_dir),
            metric.user_config.batch_size,
            device,
            execution_providers,
        )
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
        return MetricResult.parse_obj(metric_res)

    def evaluate(
        self,
        model: OliveModel,
        metrics: List[Metric],
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        metrics_res = {}
        for original_metric in metrics:
            # use model io_config if user does not specify input_names and input_shapes
            # only do this if data_config or dataloader is not provided
            # priority: dataloader_func > data_config > user_config.input_names/input_shapes > model io_config
            metric = OliveEvaluator.generate_metric_user_config_with_model_io(original_metric, model)
            dataloader, eval_func, post_func = OliveEvaluator.get_user_config(metric)

            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = self._evaluate_accuracy(
                    model, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = self._evaluate_latency(
                    model, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.CUSTOM:
                metrics_res[metric.name] = self._evaluate_custom(
                    model, metric, dataloader, eval_func, post_func, device, execution_providers
                )
            else:
                raise TypeError(f"{metric.type} is not a supported metric type")
        return flatten_metric_result(metrics_res)

    @staticmethod
    def generate_metric_user_config_with_model_io(metric: Metric, model: OliveModel):
        # if the io_config is not specified in the metrics, use the one in the model
        # should not change the original metric object which is created from config jsons
        # otherwise, if affects hashing + caching of the olive restoring.
        metric = deepcopy(metric)
        if metric.data_config:
            return metric

        io_config = model.get_io_config()
        if not io_config:
            return metric

        if not is_io_config_static(io_config):
            logger.debug(
                "Model input shapes are not static. Cannot use inferred input shapes for creating dummy data. This will"
                " cause an error when creating dummy data for tuning."
            )
        if io_config and not metric.user_config.input_names and not metric.user_config.input_shapes:
            # input_types is optional
            metric.user_config.input_names = io_config["input_names"]
            metric.user_config.input_shapes = io_config["input_shapes"]
            metric.user_config.input_types = io_config.get("input_types")
        return metric

    @staticmethod
    def get_user_config(metric: Metric):
        user_module = UserModuleLoader(metric.user_config.user_script, metric.user_config.script_dir)

        post_processing_func = getattr(metric.user_config, "post_processing_func", None)
        post_func = user_module.load_object(post_processing_func)

        dataloader_func = getattr(metric.user_config, "dataloader_func", None)
        dataloader = user_module.call_object(
            dataloader_func, get_local_path(metric.user_config.data_dir), metric.user_config.batch_size
        )

        evaluate_func = getattr(metric.user_config, "evaluate_func", None)
        eval_func = user_module.load_object(evaluate_func)

        if (not dataloader or not post_func) and metric.data_config:
            dc = metric.data_config.to_data_container()

            # TODO remove user_scripts dataloader: we should respect user scripts
            # dataloder to meet back compatibility for time being.
            dataloader = dataloader or dc.create_dataloader()
            post_func = post_func or dc.config.post_process

        if metric.user_config.input_names and metric.user_config.input_shapes and not dataloader and not eval_func:
            dataloader = (
                data_config_template.dummy_data_config_template(
                    input_names=metric.user_config.input_names,
                    input_shapes=metric.user_config.input_shapes,
                    input_types=metric.user_config.input_types,
                )
                .to_data_container()
                .create_dataloader()
            )

        return dataloader, eval_func, post_func

    @staticmethod
    def compute_accuracy(metric: Metric, preds: Any, targets: Any) -> MetricResult:
        """
        Compute accuracy metrics
        """
        evaluate_backend_cls = MetricBackend.registry[metric.backend]
        return evaluate_backend_cls().measure(preds, targets, metric)

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
            metric_res[sub_type.name] = SubMetricResult(
                value=latency_metrics[sub_type.name],
                priority=sub_type.priority,
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

    def _evaluate_onnx_accuracy(
        self,
        model: ONNXModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        session = model.prepare_session(
            inference_settings=self.get_inference_settings(metric),
            device=device,
            execution_providers=execution_providers,
        )
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

    def _evaluate_onnx_latency(
        self,
        model: OliveModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)

        session = model.prepare_session(
            inference_settings=self.get_inference_settings(metric),
            device=device,
            execution_providers=execution_providers,
        )
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

    @staticmethod
    def _evaluate_distributed_accuracy_worker(config) -> Tuple[List[Any], List[Any]]:
        model_path = config["model_path"]
        local_rank = config["local_rank"]
        world_size = config["world_size"]
        inference_settings = config.get("inference_settings", {}) or {}
        metric = Metric.from_json(config["metric"])
        dataloader, _, post_func = OnnxEvaluator.get_user_config(metric)

        import os

        os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

        from mpi4py import MPI

        local_rank = MPI.COMM_WORLD.Get_rank()

        # TODO: EPs should be selected based on accelerator_spec param passed down from the engine
        inference_settings["execution_provider"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        inference_settings["provider_options"] = [{"device_id": str(local_rank)}, {}]

        model = ONNXModel(model_path, inference_settings=inference_settings)
        session = model.prepare_session(inference_settings=inference_settings, device=Device.GPU, rank=int(local_rank))
        io_config = model.get_io_config()

        preds = []
        targets = []
        output_names = io_config["output_names"]
        for _, (input_data, labels) in enumerate(dataloader):
            input_dict = OnnxEvaluator.format_input(input_data, io_config)
            MPI.COMM_WORLD.barrier()  # Synchronize before starting each run
            output = session.run(input_feed=input_dict, output_names=None)
            output = torch.Tensor(output[0]) if len(output_names) == 1 else torch.Tensor(output)
            output = post_func(output) if post_func else output
            preds.extend(output.tolist())
            targets.extend(labels.data.tolist())

        return preds, targets

    def _evaluate_distributed_accuracy(self, model: DistributedOnnxModel, metric: Metric) -> MetricResult:
        from copy import deepcopy

        from mpi4py.futures import MPIPoolExecutor

        config = {
            "model_path": None,
            "local_rank": None,
            "world_size": model.ranks,
            "inference_settings": self.get_inference_settings(metric),
            "metric": metric.to_json(),
        }

        args = []
        for rank in range(model.ranks):
            cfg = deepcopy(config)
            cfg["local_rank"] = rank
            cfg["model_path"] = model.ranked_model_path(rank)
            args.append(cfg)

        with MPIPoolExecutor(max_workers=model.ranks) as executor:
            results = executor.map(OnnxEvaluator._evaluate_distributed_accuracy_worker, args)
            executor.shutdown()

        preds = [x for p, _ in results for x in p]
        targets = [x for _, t in results for x in t]
        return OliveEvaluator.compute_accuracy(metric, preds, targets)

    @staticmethod
    def _evaluate_distributed_latency_worker(config) -> List[float]:
        model_path = config["model_path"]
        local_rank = config["local_rank"]
        world_size = config["world_size"]
        inference_settings = config.get("inference_settings", {}) or {}
        metric = Metric.from_json(config["metric"])
        dataloader, _, _ = OnnxEvaluator.get_user_config(metric)

        import os

        os.environ["OMPI_COMM_WORLD_RANK"] = str(local_rank)
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

        from mpi4py import MPI

        local_rank = MPI.COMM_WORLD.Get_rank()
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        # TODO: EPs should be selected based on accelerator_spec param passed down from the engine
        inference_settings["execution_provider"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        inference_settings["provider_options"] = [{"device_id": str(local_rank)}, {}]

        model = ONNXModel(model_path, inference_settings=inference_settings)
        session = model.prepare_session(inference_settings=inference_settings, device=Device.GPU, rank=int(local_rank))
        io_config = model.get_io_config()

        input_feed, _ = next(iter(dataloader))
        input_feed = OnnxEvaluator.format_input(input_feed, io_config)

        if metric.user_config.io_bind:
            io_bind_op = session.io_binding()
            for k, v in input_feed.items():
                io_bind_op.bind_cpu_input(k, v)
            for item in session.get_outputs():
                io_bind_op.bind_output(item.name, "cuda")

        latencies = []
        for i in range(warmup_num + repeat_test_num):
            MPI.COMM_WORLD.barrier()  # Synchronize before starting each run
            start_time = time.perf_counter()
            if metric.user_config.io_bind:
                session.run_with_iobinding(io_bind_op)
            else:
                session.run(input_feed=input_feed, output_names=None)
            if i > warmup_num:
                latencies.append(time.perf_counter() - start_time)
            time.sleep(sleep_num)

        return latencies

    def _evaluate_distributed_latency(self, model: DistributedOnnxModel, metric: Metric) -> MetricResult:
        from copy import deepcopy

        from mpi4py.futures import MPIPoolExecutor

        config = {
            "model_path": None,
            "local_rank": None,
            "world_size": model.ranks,
            "inference_settings": self.get_inference_settings(metric),
            "metric": metric.to_json(),
        }

        args = []
        for rank in range(model.ranks):
            cfg = deepcopy(config)
            cfg["local_rank"] = rank
            cfg["model_path"] = model.ranked_model_path(rank)
            args.append(cfg)

        with MPIPoolExecutor(max_workers=model.ranks) as executor:
            results = executor.map(OnnxEvaluator._evaluate_distributed_latency_worker, args)
            executor.shutdown()

        latencies = [x for r in results for x in r]
        return OliveEvaluator.compute_latency(metric, latencies)

    def _evaluate_accuracy(
        self,
        model: ONNXModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        if isinstance(model, ONNXModel):
            return self._evaluate_onnx_accuracy(model, metric, dataloader, post_func, device, execution_providers)
        elif isinstance(model, DistributedOnnxModel):
            return self._evaluate_distributed_accuracy(model, metric)
        else:
            raise TypeError(f"Cannot evaluate accuracy for model of type: {type(model)}")

    def _evaluate_latency(
        self,
        model: OliveModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        if isinstance(model, ONNXModel):
            return self._evaluate_onnx_latency(model, metric, dataloader, post_func, device, execution_providers)
        elif isinstance(model, DistributedOnnxModel):
            return self._evaluate_distributed_latency(model, metric)
        else:
            raise TypeError(f"Cannot evaluate latency for model of type: {type(model)}")


class PyTorchEvaluator(OliveEvaluator, framework=Framework.PYTORCH):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _device_string_to_torch_device(device: Device):
        return torch.device("cuda") if device == Device.GPU else torch.device(device)

    def _evaluate_accuracy(
        self,
        model: PyTorchModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
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
        self,
        model: PyTorchModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        warmup_num, repeat_test_num, _ = get_latency_config_from_metric(metric)
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
        self,
        model: SNPEModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        dataloader = self._prepare_dataloader(dataloader, model)
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
        self,
        model: SNPEModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        dataloader = self._prepare_dataloader(dataloader, model)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session = model.prepare_session(inference_settings=self.get_inference_settings(metric), device=device)

        data_dir, input_data, _ = next(iter(dataloader))
        total_runs = warmup_num + repeat_test_num
        results = session(input_data, data_dir, runs=total_runs, sleep=sleep_num)
        latencies = results["latencies"]["total_inference_time"][warmup_num:]

        return OliveEvaluator.compute_latency(metric, latencies)

    def _prepare_dataloader(self, dataloader: Dataset, model: SNPEModel) -> SNPEDataLoader:
        if isinstance(dataloader, SNPEDataLoader):
            return dataloader
        return SNPECommonDataLoader(dataloader, model.io_config)


class OpenVINOEvaluator(OliveEvaluator, framework=Framework.OPENVINO):
    def __init__(self):
        super().__init__()

    def _evaluate_accuracy(
        self,
        model: OpenVINOModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
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
        self,
        model: OpenVINOModel,
        metric: Metric,
        dataloader: Dataset,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
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

        metric_names = set([metric.name for metric in v])
        assert len(metric_names) == metric_len, "Metric names must be unique"

        sub_type_names = set()
        sub_type_with_rank = set()
        rank_set = set()
        for metric in v:
            for sub_type in metric.sub_types:
                sub_type_names.add(joint_metric_key(metric.name, sub_type.name))
                if sub_type.priority != -1:
                    sub_type_with_rank.add(sub_type.name)
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
            raise ValueError(f"Priorities must be unique and in the range 1 to {metric_len}")

        return v
