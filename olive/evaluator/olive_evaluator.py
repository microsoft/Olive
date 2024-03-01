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
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, NamedTuple, Tuple, Type, Union

import numpy as np
import torch

import olive.data.template as data_config_template
from olive.cache import get_local_path_from_root
from olive.common.config_utils import ConfigBase
from olive.common.ort_inference import OrtInferenceSession, prepare_io_bindings
from olive.common.pydantic_v1 import validator
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import tensor_data_to_device
from olive.constants import Framework
from olive.evaluator.metric import (
    LatencySubType,
    Metric,
    MetricResult,
    MetricType,
    SubMetricResult,
    ThroughputSubType,
    flatten_metric_result,
    get_latency_config_from_metric,
    joint_metric_key,
)
from olive.evaluator.metric_backend import MetricBackend
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
    registry: ClassVar[Dict[str, Type["OliveEvaluator"]]] = {}

    @classmethod
    def __init_subclass__(cls, framework: Framework, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.framework = framework
        cls.registry[str(framework).lower()] = cls

    @classmethod
    def io_bind_enabled(cls, metric: Metric, inference_settings: Dict) -> bool:
        if metric.user_config.io_bind:
            return True

        if inference_settings and inference_settings.get("io_bind"):
            return True

        return False

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
        data_root: str,
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
        data_root: str,
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
        data_root: str,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        latencies = self._evaluate_raw_latency(
            model, data_root, metric, dataloader, post_func, device, execution_providers
        )
        return OliveEvaluator.compute_latency(metric, latencies)

    def _evaluate_throughput(
        self,
        model: "OliveModelHandler",
        data_root: str,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> MetricResult:
        latencies = self._evaluate_raw_latency(
            model, data_root, metric, dataloader, post_func, device, execution_providers
        )
        return OliveEvaluator.compute_throughput(metric, latencies)

    def _evaluate_custom(
        self,
        model: "OliveModelHandler",
        data_root: str,
        metric: Metric,
        dataloader: "DataLoader",
        eval_func,
        post_func=None,
        device: Device = Device.CPU,
        execution_providers=None,
    ) -> MetricResult:
        raw_res = None
        if metric.user_config.evaluate_func:
            raw_res = eval_func(
                model,
                get_local_path_from_root(data_root, metric.user_config.data_dir),
                metric.user_config.batch_size,
                device,
                execution_providers,
            )
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
        data_root: str,
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
            dataloader, eval_func, post_func = OliveEvaluator.get_user_config(model.framework, data_root, metric)
            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = self._evaluate_accuracy(
                    model, data_root, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = self._evaluate_latency(
                    model, data_root, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.THROUGHPUT:
                metrics_res[metric.name] = self._evaluate_throughput(
                    model, data_root, metric, dataloader, post_func, device, execution_providers
                )
            elif metric.type == MetricType.CUSTOM:
                metrics_res[metric.name] = self._evaluate_custom(
                    model, data_root, metric, dataloader, eval_func, post_func, device, execution_providers
                )
            else:
                raise TypeError(f"{metric.type} is not a supported metric type")
        return flatten_metric_result(metrics_res)

    @staticmethod
    def generate_metric_user_config_with_model_io(metric: Metric, model: "OliveModelHandler"):
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
            metric.user_config.input_names = io_config["input_names"]
            # input_shapes is optional for hf models
            metric.user_config.input_shapes = io_config.get("input_shapes")
            # input_types is optional which can be None. If None, it will be replaced with float32 in DummyDataset
            metric.user_config.input_types = io_config.get("input_types")
        return metric

    @staticmethod
    def _get_func_kwargs(metric: Metric, func_name: str):
        """Get the function kwargs from the metric config."""
        if metric.user_config.func_kwargs:
            return metric.user_config.func_kwargs.get(func_name, {})
        return {}

    @classmethod
    def get_user_config(cls, framework: Framework, data_root: str, metric: Metric):
        assert metric.user_config, "user_config is not specified in the metric config"
        user_module = UserModuleLoader(metric.user_config.user_script, metric.user_config.script_dir)

        # load the post processing function
        post_processing_func = getattr(metric.user_config, "post_processing_func", None)
        post_func = user_module.load_object(post_processing_func)
        post_func_kwargs = cls._get_func_kwargs(metric, "post_processing_func")
        if post_func_kwargs:
            # apply the kwargs to the post processing function
            post_func = partial(post_func, **post_func_kwargs)

        # load the dataloader function and create the dataloader
        dataloader_func = getattr(metric.user_config, "dataloader_func", None)
        if dataloader_func:
            data_dir = get_local_path_from_root(data_root, metric.user_config.data_dir)
            dataloader = user_module.call_object(
                dataloader_func,
                data_dir,
                metric.user_config.batch_size,
                model_framework=framework,
                **cls._get_func_kwargs(metric, "dataloader_func"),
            )
        else:
            dataloader = None

        # load the evaluate function
        # priority: evaluate_func > metric_func
        eval_func = None
        if metric.type == MetricType.CUSTOM:
            evaluate_func = getattr(metric.user_config, "evaluate_func", None)
            kwargs = cls._get_func_kwargs(metric, "evaluate_func")
            if not evaluate_func:
                evaluate_func = getattr(metric.user_config, "metric_func", None)
                kwargs = cls._get_func_kwargs(metric, "metric_func")

            if not evaluate_func:
                raise ValueError("evaluate_func or metric_func is not specified in the metric config")

            eval_func = user_module.load_object(evaluate_func)
            if kwargs:
                eval_func = partial(eval_func, **kwargs)

        # get dataloader and/or post processing function from data_config if not specified in the metric config
        if (not dataloader or not post_func) and metric.data_config:
            dc = metric.data_config.to_data_container()

            # TODO(trajep): remove user_scripts dataloader: we should respect user scripts
            # dataloder to meet back compatibility for time being.
            dataloader = dataloader or dc.create_dataloader(data_root)
            post_func = post_func or dc.config.post_process

        # get dataloader and/or post processing function from model io_config if not specified in the metric config
        # or data config
        if metric.user_config.input_names and metric.user_config.input_shapes and not dataloader and not eval_func:
            dataloader = (
                data_config_template.dummy_data_config_template(
                    input_names=metric.user_config.input_names,
                    input_shapes=metric.user_config.input_shapes,
                    input_types=metric.user_config.input_types,
                )
                .to_data_container()
                .create_dataloader(data_root)
            )

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
        batch_size = metric.user_config.batch_size
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


class OnnxEvaluatorMixin:

    @staticmethod
    def format_input(input_data, io_config):
        """Format input data to ONNX input format."""
        input_names = io_config["input_names"]
        name_to_type = dict(zip(io_config["input_names"], io_config["input_types"]))
        if isinstance(input_data, list):
            input_data = dict(zip(input_names, input_data))
        elif not isinstance(input_data, dict):
            input_data = dict(zip(input_names, [input_data]))
        return {
            k: np.ascontiguousarray(
                input_data[k].cpu().numpy() if isinstance(input_data[k], torch.Tensor) else input_data[k],
                dtype=name_to_type[k],
            )
            for k in input_data
            if k in input_names
        }

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


class OnnxEvaluator(OliveEvaluator, OnnxEvaluatorMixin, framework=Framework.ONNX):

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
        io_config = model.get_io_config()
        io_bind = OnnxEvaluator.io_bind_enabled(metric, model.inference_settings)
        shared_kv_buffer = metric.user_config.shared_kv_buffer
        use_fp16 = any(v == "float16" for v in io_config["input_types"])
        input_feed = None
        if io_bind and shared_kv_buffer and use_fp16:
            input_feed = OnnxEvaluator.format_input(next(iter(dataloader))[0], io_config)

        # create session wrapper
        session_wrapper = OrtInferenceSession(
            session,
            io_bind=io_bind,
            device=device,
            shared_kv_buffer=shared_kv_buffer,
            use_fp16=use_fp16,
            input_feed=input_feed,
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
        io_config = model.get_io_config()

        preds = []
        targets = []
        logits = []
        logits_dict = collections.defaultdict(list)
        output_names = io_config["output_names"]
        is_single_tensor_output = len(output_names) == 1
        for input_data, labels in dataloader:
            input_feed = OnnxEvaluator.format_input(input_data, io_config)
            result = session.run(input_feed)
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
            logits = {k: torch.cat(logits[k], dim=0) for k in output_names}

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
        io_config = model.get_io_config()

        input_data, _ = next(iter(dataloader))
        input_feed = OnnxEvaluator.format_input(input_data, io_config)

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
        data_root = config["data_root"]
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
        dataloader, _, post_func = OnnxEvaluator.get_user_config(model.framework, data_root, metric)

        session = model.prepare_session(inference_settings=inference_settings, device=Device.GPU, rank=int(local_rank))
        io_config = model.get_io_config()

        preds = []
        targets = []
        logits = []
        output_names = io_config["output_names"]
        for _, (input_data, labels) in enumerate(dataloader):
            input_dict = OnnxEvaluator.format_input(input_data, io_config)
            MPI.COMM_WORLD.barrier()  # Synchronize before starting each run
            output = session.run(input_feed=input_dict, output_names=None)
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
        data_root: str,
        metric: Metric,
        device: Device,
        execution_providers: Union[str, List[str]],
    ) -> MetricResult:
        from mpi4py.futures import MPIPoolExecutor

        config = {
            "model_path": None,
            "local_rank": None,
            "world_size": model.num_ranks,
            "inference_settings": metric.get_inference_settings(self.framework.lower()),
            "metric": metric.to_json(),
        }

        args = []
        for rank in range(model.num_ranks):
            cfg = deepcopy(config)
            cfg["local_rank"] = rank
            cfg["model_path"] = model.ranked_model_path(rank)
            cfg["data_root"] = data_root
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
    def _evaluate_distributed_latency_worker(data_root, config) -> List[float]:
        model_path = config["model_path"]
        data_root = config["data_root"]
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
        dataloader, _, _ = OnnxEvaluator.get_user_config(model.framework, data_root, metric)
        session = model.prepare_session(inference_settings=inference_settings, device=Device.GPU, rank=int(local_rank))
        io_config = model.get_io_config()

        input_feed, _ = next(iter(dataloader))
        input_feed = OnnxEvaluator.format_input(input_feed, io_config)
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
                session.run(input_feed=input_feed, output_names=None)
            if i > warmup_num:
                latencies.append(time.perf_counter() - start_time)
            time.sleep(sleep_num)

        return latencies

    def _evaluate_distributed_latency(
        self,
        model: DistributedOnnxModelHandler,
        data_root: str,
        metric: Metric,
        device,
        execution_providers: Union[str, List[str]],
    ) -> List[float]:
        from mpi4py.futures import MPIPoolExecutor

        config = {
            "model_path": None,
            "local_rank": None,
            "world_size": model.num_ranks,
            "inference_settings": metric.get_inference_settings(self.framework.lower()),
            "metric": metric.to_json(),
        }

        args = []
        for rank in range(model.num_ranks):
            cfg = deepcopy(config)
            cfg["local_rank"] = rank
            cfg["model_path"] = model.ranked_model_path(rank)
            cfg["data_root"] = data_root
            cfg["device"] = device
            cfg["providers"] = execution_providers
            args.append(cfg)

        with MPIPoolExecutor(max_workers=model.num_ranks) as executor:
            results = executor.map(OnnxEvaluator._evaluate_distributed_latency_worker, args)
            executor.shutdown()

        return [x for r in results for x in r]

    def _evaluate_accuracy(
        self,
        model: ONNXModelHandler,
        data_root: str,
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
            return self._evaluate_distributed_accuracy(model, data_root, metric, device, execution_providers)
        else:
            raise TypeError(f"Cannot evaluate accuracy for model of type: {type(model)}")

    def _evaluate_raw_latency(
        self,
        model: "OliveModelHandler",
        data_root: str,
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
            return self._evaluate_distributed_latency(model, data_root, metric, device, execution_providers)
        else:
            raise TypeError(f"Cannot evaluate latency for model of type: {type(model)}")


class PyTorchEvaluator(OliveEvaluator, framework=Framework.PYTORCH):

    @staticmethod
    def _device_string_to_torch_device(device: Device):
        return torch.device("cuda") if device == Device.GPU else torch.device(device)

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
        device = PyTorchEvaluator._device_string_to_torch_device(device)
        if device:
            session.to(device)
        for input_data_i, labels in dataloader:
            input_data = tensor_data_to_device(input_data_i, device)
            result = session(**input_data) if isinstance(input_data, dict) else session(input_data)
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
        data_root: str,
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
        data_root: str,
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
        device = PyTorchEvaluator._device_string_to_torch_device(device)
        is_cuda = device == Device.GPU
        if device:
            session.to(device)
            input_data = tensor_data_to_device(input_data, device)
        input_is_dict = isinstance(input_data, dict)

        # warm up
        for _ in range(warmup_num):
            session(**input_data) if input_is_dict else session(input_data)

        latencies = []
        if not is_cuda:
            for _ in range(repeat_test_num):
                t = time.perf_counter()
                # TODO(jambayk): do we care about the efficiency of if/else here?
                # probably won't add much overhead compared to the inference time
                # also we are doing the same for all models
                session(**input_data) if input_is_dict else session(input_data)
                latencies.append(time.perf_counter() - t)
        else:
            # synchronize before starting the test
            torch.cuda.synchronize()
            # cuda events for measuring latency
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(repeat_test_num):
                starter.record()
                session(**input_data) if input_is_dict else session(input_data)
                ender.record()
                # synchronize after forward pass
                torch.cuda.synchronize()
                # add time in seconds, originally in milliseconds
                latencies.append(starter.elapsed_time(ender) * 1e-3)

        # move model to cpu
        if device:
            session.to("cpu")
        # only move to cpu cannot release gpu memory, call cuda.empty_cache() to release gpu memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return latencies


class SNPEEvaluator(OliveEvaluator, framework=Framework.SNPE):

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
        inference_settings = metric.get_inference_settings(self.framework.lower())
        # for accuracy evaluation, the `return_numpy_results` is required to be True
        # but for model inference, it is not required to be True.
        # We just set it to True for simple evaluation.
        inference_settings["return_numpy_results"] = True

        session = model.prepare_session(inference_settings=inference_settings, device=device)

        preds = []
        targets = []
        logits = []
        for data_dir, input_list, labels in dataloader:
            result = session(input_list, data_dir)
            # as the SNPE inference will return a list of outputs which is beyond the model output shape
            # we need to squeeze the fist dimensions of output to get right accuracy metrics
            for idx, output in enumerate(result.get("results")):
                post_output = output
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
        data_root: str,
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
        data_root: str,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        dataloader = self._prepare_dataloader(dataloader, model, 1)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(self.framework.lower()), device=device
        )

        data_dir, input_data, _ = next(iter(dataloader))
        total_runs = warmup_num + repeat_test_num
        results = session(input_data, data_dir, runs=total_runs, sleep=sleep_num)
        return results["latencies"]["total_inference_time"][warmup_num:]

    def _prepare_dataloader(
        self, dataloader: Union["DataLoader", FileListDataLoader], model: "SNPEModelHandler", file_chunk_size=None
    ) -> FileListDataLoader:
        if isinstance(dataloader, FileListDataLoader):
            return dataloader
        return FileListCommonDataLoader(dataloader, model.io_config, batch_size=file_chunk_size)


class OpenVINOEvaluator(OliveEvaluator, framework=Framework.OPENVINO):

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
            inference_settings=metric.get_inference_settings(self.framework.lower()), device=device
        )

        preds = []
        targets = []
        logits = []
        for input_data, label in dataloader:
            session.infer({0: input_data})
            result = session.get_output_tensor(0).data
            outputs = post_func(result) if post_func else result
            preds.extend(outputs)
            targets.extend(label)
            logits.extend(result)
        return OliveModelOutput(preds=preds, logits=logits), targets

    def _evaluate_accuracy(
        self,
        model: "OpenVINOModelHandler",
        data_root: str,
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
        data_root: str,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(self.framework.lower()), device=device
        )

        latencies = []
        for input_data, _ in dataloader:
            t = time.perf_counter()
            session.infer(input_data)
            latencies.append(time.perf_counter() - t)
        return latencies


class QNNEvaluator(OliveEvaluator, framework=Framework.QNN):

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
            inference_settings=metric.get_inference_settings(self.framework.lower()), device=device
        )

        preds = []
        targets = []
        logits = []
        for data_dir, input_list, labels in dataloader:
            result = session(input_list, data_dir)
            for idx, output in enumerate(result.get("result")):
                post_output = output
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
        data_root: str,
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
        data_root: str,
        metric: Metric,
        dataloader: "DataLoader",
        post_func=None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
    ) -> List[float]:
        dataloader = self._prepare_dataloader(dataloader, model, 1)
        warmup_num, repeat_test_num, sleep_num = get_latency_config_from_metric(metric)
        session = model.prepare_session(
            inference_settings=metric.get_inference_settings(self.framework.lower()), device=device
        )

        data_dir, input_data, _ = next(iter(dataloader))
        # for qnn-net-run only keep 20 logs
        total_runs = min(warmup_num + repeat_test_num, 20)
        results = session(input_data, data_dir, runs=total_runs, sleep=sleep_num)
        return results["latencies"]["net_run"][warmup_num:]

    def _prepare_dataloader(
        self, dataloader: "DataLoader", model: "QNNModelHandler", file_chunk_size=None
    ) -> FileListDataLoader:
        if isinstance(dataloader, FileListDataLoader):
            return dataloader
        return FileListCommonDataLoader(dataloader, model.io_config, batch_size=file_chunk_size)


class OliveEvaluatorFactory:
    @staticmethod
    def create_evaluator_for_model(model: "OliveModelHandler") -> OliveEvaluator:
        evaluator_cls = OliveEvaluator.registry[str(model.framework).lower()]
        return evaluator_cls()


class OliveEvaluatorConfig(ConfigBase):
    metrics: List[Metric] = []  # noqa: RUF012

    @property
    def is_accuracy_drop_tolerance(self):
        for metric in self.metrics:
            for sub_metric in metric.sub_types:
                if metric.type == MetricType.ACCURACY and sub_metric.higher_is_better:
                    return sub_metric.goal is not None and sub_metric.goal.has_regression_goal()
        return False

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
            raise ValueError(f"Priorities must be unique and in the range 1 to {metric_len}")

        return v
