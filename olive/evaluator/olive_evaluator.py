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
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch

from olive.common.config_utils import NestedConfig, validate_config
from olive.common.import_lib import import_user_module
from olive.common.ort_inference import OrtInferenceSession, prepare_io_bindings
from olive.common.pydantic_v1 import Field, root_validator, validator
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import format_data, load_weights, tensor_data_to_device
from olive.constants import Framework
from olive.data.config import DataConfig
from olive.data.container.dummy_data_container import TRANSFORMER_DUMMY_DATA_CONTAINER
from olive.data.template import dummy_data_config_template
from olive.evaluator.metric import LatencySubType, Metric, MetricType, ThroughputSubType, get_latency_config_from_metric
from olive.evaluator.metric_backend import MetricBackend
from olive.evaluator.metric_result import MetricResult, SubMetricResult, flatten_metric_result, joint_metric_key
from olive.evaluator.registry import Registry
from olive.hardware import Device
from olive.model import DistributedOnnxModelHandler, ONNXModelHandler
from olive.model.config.io_config import is_io_config_static
from olive.model.utils.onnx_utils import dump_tuning_result
from olive.platform_sdk.qualcomm.utils.data_loader import FileListCommonDataLoader, FileListDataLoader

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from olive.model import (
        OliveModelHandler,
        OpenVINOModelHandler,
        PyTorchModelHandler,
        QNNModelHandler,
        SNPEModelHandler,
    )

logger = logging.getLogger(__name__)

# pylint: disable=useless-parent-delegation


class OliveModelOutput(NamedTuple):
    preds: Any
    logits: Any


class OliveEvaluator(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: List[Metric],
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
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
        assert metric.user_config, "user_config is not specified in the metric config"

        dataloader = None
        eval_func = None
        post_func = None

        # load the evaluate function
        # priority: evaluate_func > metric_func
        if metric.type == MetricType.CUSTOM:
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
    def compute_accuracy(metric: Metric, model_outputs: Union[Tuple, NamedTuple], targets: Any) -> MetricResult:
        """Compute accuracy metrics."""
        evaluate_backend_cls = MetricBackend.registry[metric.backend]
        return evaluate_backend_cls().measure(model_outputs, targets, metric)

    @staticmethod
    def latency_helper(latencies) -> Dict:
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
        return MetricResult.parse_obj(metric_res)

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
        return MetricResult.parse_obj(metric_res)


class _OliveEvaluator(OliveEvaluator):
    @staticmethod
    def device_string_to_torch_device(device: Device):
        return torch.device("cuda") if device == Device.GPU else torch.device(device)

    @classmethod
    def io_bind_enabled(cls, metric: Metric, inference_settings: Dict) -> bool:
        if metric.user_config.io_bind:
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
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
        raise NotImplementedError

    @abstractmethod
    def _evaluate_accuracy(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
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
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        """For given repeat_test_num, return a list of latencies(ms)."""
        raise NotImplementedError

    def _evaluate_latency(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        latencies = self._evaluate_raw_latency(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_latency(metric, latencies)

    def _evaluate_throughput(
        self,
        model: "OliveModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        latencies = self._evaluate_raw_latency(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_throughput(metric, latencies)

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
        if metric.user_config.evaluate_func:
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
        return MetricResult.parse_obj(metric_res)

    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: List[Metric],
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
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
            elif metric.type == MetricType.CUSTOM:
                metrics_res[metric.name] = self._evaluate_custom(
                    model, metric, dataloader, eval_func, post_func, device, execution_providers
                )
            else:
                raise TypeError(f"{metric.type} is not a supported metric type")
        return flatten_metric_result(metrics_res)


class OnnxEvaluatorMixin:

    @staticmethod
    def get_inference_settings(metric: Metric, model: ONNXModelHandler) -> Dict[str, Any]:
        # user.config.inference_settings > model.inference_settings > default inference_settings
        # when user.config.inference_settings is None, the model.inference_settings
        # will be used in model.prepare_session(..)
        inference_settings = {}
        model_infrerence_settings = model.inference_settings
        if model_infrerence_settings:
            inference_settings.update(model_infrerence_settings)

        metric_inference_settings = metric.get_inference_settings(Framework.ONNX.lower())
        if metric_inference_settings:
            inference_settings.update(metric_inference_settings)

        return inference_settings


@Registry.register(str(Framework.ONNX))
@Registry.register("OnnxEvaluator")
class OnnxEvaluator(_OliveEvaluator, OnnxEvaluatorMixin):

    @staticmethod
    def get_session_wrapper(
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        device: Device,
        execution_providers: List[str],
    ) -> Tuple[OrtInferenceSession, Dict[str, Any]]:
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
        shared_kv_buffer = metric.user_config.shared_kv_buffer
        use_fp16 = any(v == "float16" for v in io_config["input_types"])
        input_feed = None
        if io_bind and shared_kv_buffer and use_fp16:
            input_feed = format_data(next(iter(dataloader))[0], io_config)

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
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
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
        for input_data, labels in dataloader:
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

    def _evaluate_onnx_accuracy(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        inference_output, targets = self._inference(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    def _evaluate_onnx_latency(
        self,
        model: ONNXModelHandler,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session, inference_settings = OnnxEvaluator.get_session_wrapper(
            model, metric, dataloader, device, execution_providers
        )
        io_config = model.io_config

        input_data, _ = next(iter(dataloader))
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
    def _evaluate_distributed_accuracy_worker(config) -> Tuple[List[Any], List[Any]]:
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
        for _, (input_data, labels) in enumerate(dataloader):
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
        execution_providers: Union[str, List[str]],
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
    def _evaluate_distributed_latency_worker(config) -> List[float]:
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

        input_feed, _ = next(iter(dataloader))
        input_feed = format_data(input_feed, io_config)
        kv_cache_ortvalues = {} if metric.user_config.shared_kv_buffer else None

        io_bind = OnnxEvaluator.io_bind_enabled(metric, model.inference_settings)
        if io_bind:
            io_bind_op = prepare_io_bindings(
                session,
                input_feed,
                Device.GPU,
                shared_kv_buffer=metric.user_config.shared_kv_buffer,
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
        execution_providers: Union[str, List[str]],
    ) -> List[float]:
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
        execution_providers: Union[str, List[str]] = None,
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
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
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
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
        session = model.prepare_session()
        preds = []
        targets = []
        logits = []
        device = _OliveEvaluator.device_string_to_torch_device(device)
        run_kwargs = metric.get_run_kwargs()
        if device:
            session.to(device)
        for input_data_i, labels in dataloader:
            input_data = tensor_data_to_device(input_data_i, device)
            result = model.run_session(session, input_data, **run_kwargs)
            outputs = post_func(result) if post_func else result
            # keep the outputs and results as torch tensor on cpu
            # it is expensive to convert to list and then convert back to torch tensor
            preds.append(outputs.cpu())
            targets.append(labels.cpu())
            logits.append(
                result.logits.cpu()
                if not isinstance(result, torch.Tensor) and getattr(result, "logits", None) is not None
                else result.cpu()
            )
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
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        inference_output, targets = self._inference(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    @torch.no_grad()
    def _evaluate_raw_latency(
        self,
        model: "PyTorchModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        # pylint: disable=expression-not-assigned
        warmup_num, repeat_test_num, _ = get_latency_config_from_metric(metric)
        # pytorch model doesn't use inference_settings, so we can pass None
        session = model.prepare_session(inference_settings=None, device=device)

        input_data, _ = next(iter(dataloader))
        torch_device = _OliveEvaluator.device_string_to_torch_device(device)
        run_kwargs = metric.get_run_kwargs()

        is_cuda = device == Device.GPU
        if is_cuda:
            session.to(torch_device)
            input_data = tensor_data_to_device(input_data, torch_device)

        # warm up
        for _ in range(warmup_num):
            model.run_session(session, input_data, **run_kwargs)

        latencies = []
        if is_cuda:
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


@Registry.register(str(Framework.SNPE))
@Registry.register("SNPEEvaluator")
class SNPEEvaluator(_OliveEvaluator):

    def _inference(
        self,
        model: "SNPEModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
        dataloader = self._prepare_dataloader(dataloader, model)
        inference_settings = metric.get_inference_settings(Framework.SNPE.lower())
        # for accuracy evaluation, the `return_numpy_results` is required to be True
        # but for model inference, it is not required to be True.
        # We just set it to True for simple evaluation.
        inference_settings["return_numpy_results"] = True

        session = model.prepare_session(inference_settings=inference_settings, device=device)
        run_kwargs = metric.get_run_kwargs()

        preds = []
        targets = []
        logits = []
        for data_dir, input_list, labels in dataloader:
            run_kwargs["data_dir"] = data_dir
            result = model.run_session(session, input_list, **run_kwargs)
            # as the SNPE inference will return a list of outputs which is beyond the model output shape
            # we need to squeeze the fist dimensions of output to get right accuracy metrics
            for idx, output in enumerate(result.get("results")):
                if post_func:
                    post_output = post_func(output)
                else:
                    raise ValueError("Post processing function is required for SNPE model")
                preds.extend(post_output.tolist())
                if isinstance(labels[idx], (list, np.ndarray)):
                    targets.extend(labels[idx])
                else:
                    targets.append(labels[idx])
                # only when return_numpy_results is True, the result is a dict with "logits" key
                logits.extend(output.get("logits", np.array([])).tolist())
        return OliveModelOutput(preds=preds, logits=logits), targets

    def _evaluate_accuracy(
        self,
        model: "SNPEModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        inference_output, targets = self._inference(model, metric, dataloader, post_func, device, execution_providers)
        return OliveEvaluator.compute_accuracy(metric, inference_output, targets)

    def _evaluate_raw_latency(
        self,
        model: "SNPEModelHandler",
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        dataloader = self._prepare_dataloader(dataloader, model, 1)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.SNPE.lower()), device=device
        )

        data_dir, input_data, _ = next(iter(dataloader))
        total_runs = warmup_num + repeat_test_num
        run_kwargs = metric.get_run_kwargs()
        run_kwargs["data_dir"] = data_dir
        run_kwargs["runs"] = total_runs
        run_kwargs["sleep"] = sleep_num

        results = model.run_session(session, input_data, **run_kwargs)
        return results["latencies"]["total_inference_time"][warmup_num:]

    def _prepare_dataloader(
        self, dataloader: Union["DataLoader", FileListDataLoader], model: "SNPEModelHandler", file_chunk_size=None
    ) -> FileListDataLoader:
        if isinstance(dataloader, FileListDataLoader):
            return dataloader
        return FileListCommonDataLoader(dataloader, model.io_config, batch_size=file_chunk_size)


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
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.OPENVINO.lower()), device=device
        )
        run_kwargs = metric.get_run_kwargs()

        preds = []
        targets = []
        logits = []
        for input_data, label in dataloader:
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
        execution_providers: Union[str, List[str]] = None,
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
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.OPENVINO.lower()), device=device
        )
        run_kwargs = metric.get_run_kwargs()

        latencies = []
        for input_data, _ in dataloader:
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
        execution_providers: Union[str, List[str]] = None,
    ) -> Tuple[OliveModelOutput, Any]:
        dataloader = self._prepare_dataloader(dataloader, model)
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(Framework.QNN.lower()), device=device
        )

        preds = []
        targets = []
        logits = []
        run_kwargs = metric.get_run_kwargs()
        for data_dir, input_list, labels in dataloader:
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
        execution_providers: Union[str, List[str]] = None,
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
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
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
    def __init__(self, model_class: str, tasks: List[str], **kwargs):
        super().__init__(**kwargs)

        self.model_class = model_class
        self.tasks = tasks
        self.limit = kwargs.get("limit")
        self.batch_size = kwargs.get("batch_size", 1)
        self.max_gen_toks = kwargs.get("max_gen_toks")

    def evaluate(
        self,
        model: "OliveModelHandler",
        metrics: List[Metric],
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        import lm_eval

        device = _OliveEvaluator.device_string_to_torch_device(device)
        # device = torch.device("cuda:5")
        tokenizer = model.get_hf_tokenizer()
        nn_module = model.load_model().eval().to(device)

        lmmodel = lm_eval.api.registry.get_model(self.model_class)(
            pretrained=nn_module,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            device=device,
            max_gen_toks=self.max_gen_toks,
        )

        task_manager = lm_eval.tasks.TaskManager()

        results = lm_eval.simple_evaluate(
            model=lmmodel,
            tasks=self.tasks,
            task_manager=task_manager,
            log_samples=False,
            batch_size=self.batch_size,
            device=device,
            limit=self.limit,
        )

        metrics = {}
        for task_name in sorted(results["results"].keys()):
            metric_items = sorted(results["results"][task_name].items())

            task_metrics = {}
            for mf, v in metric_items:
                if mf != "alias":
                    m, _ = mf.split(",", 1)
                    if not m.endswith("_stderr"):
                        task_metrics[m] = SubMetricResult(value=v, priority=-1, higher_is_better=True)

            metrics[task_name] = MetricResult.parse_obj(task_metrics)

        return flatten_metric_result(metrics)


class OliveEvaluatorConfig(NestedConfig):
    _nested_field_name = "type_args"

    name: str = None
    type: str = None
    type_args: Dict = Field(default_factory=dict)

    # user script to define and register the evaluator
    user_script: Union[Path, str] = None
    script_dir: Union[Path, str] = None

    metrics: List[Metric] = []  # noqa: RUF012

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

    @root_validator(pre=True)
    def validate_type(cls, values):
        if values.get("user_script"):
            import_user_module(values["user_script"], values.get("script_dir"))

        evaluator_type = values.get("type")
        if evaluator_type is not None and Registry.get(evaluator_type) is None:
            raise ValueError(f"Invalid/unknown evaluator type: {evaluator_type}")

        return values

    @validator("metrics")
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
