# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import itertools
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.common.ort_inference import check_and_normalize_provider_args
from olive.data.config import DataConfig
from olive.evaluator.metric import LatencySubType, Metric, MetricType, joint_metric_key
from olive.evaluator.metric_config import get_user_config_properties_from_metric_type
from olive.exception import EXCEPTIONS_TO_RAISE
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


PERFTUNING_BASELINE = "pretuning-baseline"


def generate_tuning_combos(config):
    import onnxruntime as ort

    providers_list = (
        config.providers_list
        if config.providers_list
        else AcceleratorLookup.get_execution_providers_for_device(config.device)
    )
    execution_mode_list = (
        config.execution_mode_list
        if config.execution_mode_list
        else [ort.ExecutionMode.ORT_SEQUENTIAL.value, ort.ExecutionMode.ORT_PARALLEL.value]
    )
    opt_level_list = config.opt_level_list if config.opt_level_list else [99]

    tuning_combos = itertools.product(providers_list, execution_mode_list, opt_level_list)
    yield from tuning_combos


def valid_config(tuning_combos, config):
    # the order of combos: "provider", "execution_mode", "ort_opt_level", "io_bind"

    # Parallel execution mode does not support the CUDA Execution Provider.
    # So ORT will make the execution mode sequential when it uses the CUDA Execution Provider.

    # if the first combo is CPUExecutionProvider, then the io_bind should not be True
    providers, _ = tuning_combos[0]
    provider = providers[0]
    if provider == "CPUExecutionProvider" and tuning_combos[3]:
        logger.info("[Ignored] Because EP is CPUExecutionProvider, the io_bind should not be True")
        return False
    if provider != "CUDAExecutionProvider" and config.enable_cuda_graph:
        logger.info("[Ignored] Because EP is not CUDAExecutionProvider, the enable_cuda_graph is ignored")
        return True
    if provider != "TensorrtExecutionProvider" and config.trt_fp16_enable:
        logger.info("[Ignored] Because EP is not TensorrtExecutionProvider, the trt_fp16_enable is ignored")
        return True
    if provider == "CUDAExecutionProvider" and config.enable_cuda_graph and tuning_combos[1] == 1:
        # Need disable the ort.ExecutionMode.ORT_PARALLEL if enable_cuda_graph is True
        # because the CUDA Graph does not support the parallel execution mode.
        # Otherwise, the following error will be thrown:
        #  self._sess.run_with_iobinding(iobinding._iobinding, run_options)
        #  RuntimeError: Error in execution: /onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:121
        #  std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(
        #       ERRTYPE, const char*, const char*, ERRTYPE, const char*, const char*, int)
        #    [with ERRTYPE = cudaError; bool THRW = true;
        #       std::conditional_t<THRW, void, onnxruntime::common::Status> = void]
        #    /onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:114
        #  std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(
        #       ERRTYPE, const char*, const char*, ERRTYPE, const char*, const char*, int)
        #    [with ERRTYPE = cudaError; bool THRW = true;
        #       std::conditional_t<THRW, void, onnxruntime::common::Status> = void]
        #    CUDA failure 901: operation failed due to a previous error during capture ;
        #       GPU=0 ; hostname=41bcb832c000001 ;
        #    file=/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_graph.cc ; line=33 ;
        #    expr=cudaStreamEndCapture(stream_, &graph_);
        # The RuntimeError doesn't impact the perf-tuning result, but it will waste the time.
        logger.warning(
            "[Ignored] Because EP is CUDAExecutionProvider, the execution_mode should not be 1 "
            "in case of enable_cuda_graph is True. Otherwise, the RuntimeError will be thrown."
        )
        return False
    return True


def populate_provider_options(execution_provider, config):
    if isinstance(execution_provider, (tuple, list)):
        assert len(execution_provider) == 2, "execution_provider should be a tuple with execution provider and options"
        provider = execution_provider[0]
        provider_options = copy.deepcopy(execution_provider[1]) or {}
    elif isinstance(execution_provider, str):
        provider = execution_provider
        provider_options = {}
    else:
        raise ValueError("execution_provider should be a tuple, list or string")

    if provider == "TensorrtExecutionProvider":
        provider_options["trt_fp16_enable"] = config.trt_fp16_enable
    elif provider == "CUDAExecutionProvider":
        provider_options["enable_cuda_graph"] = config.enable_cuda_graph

    return provider, provider_options


def generate_test_name(test_params, io_bind):
    if not test_params:
        return PERFTUNING_BASELINE

    name_list = []
    ep = test_params["execution_provider"][0]
    provider_option = test_params["provider_options"][0]
    ep_name = ep.replace("ExecutionProvider", "").lower()
    ep_names = []
    if provider_option:
        ep_names.append((ep_name, provider_option))
    else:
        ep_names.append(ep_name)
    if len(ep_names) == 1:
        ep_names = ep_names[0]

    name_list.append(ep_names)
    session_options = test_params.get("session_options")
    if session_options:
        name_list.append(session_options)
    if io_bind:
        name_list.append({"io_bind": io_bind})

    return "-".join(f"{str(i)}" for i in name_list)


def enable_rocm_op_tuning(inference_settings, input_tuning_result, tuning_result_output_dir):
    execution_providers = inference_settings["execution_provider"]
    provider_options = inference_settings["provider_options"]

    def set_rocm_provider_options(provider_options):
        # Please refer to the following links for the config or ROCMExecutionProvider
        # https://github.com/microsoft/onnxruntime/blob/71657d1eb8b0a24a4b6584d9e904506a0b4e1521/onnxruntime/core/providers/rocm/rocm_execution_provider_info.cc#L24C1-L25
        provider_options["tunable_op_enable"] = True
        provider_options["tunable_op_tuning_enable"] = True
        if "device_id" not in provider_options:
            provider_options["device_id"] = 0

        return provider_options

    def find_tuning_op_result_by_ep(tuning_result, ep):
        for tuning_op_result in tuning_result:
            if tuning_op_result["ep"] == ep:
                return tuning_op_result
        return None

    assert input_tuning_result is None or isinstance(input_tuning_result, list), "tuning_result should be a list"

    tuning_result_file_name = None
    tuning_op_result = []
    for i, ep in enumerate(execution_providers):
        if ep == "ROCMExecutionProvider":
            set_rocm_provider_options(provider_options[i])
            tuning_result_file_name = "tuning_result.json"
        if input_tuning_result:
            op_result = find_tuning_op_result_by_ep(input_tuning_result, ep)
            if op_result:
                tuning_op_result.append(op_result)

    inference_settings["provider_options"] = provider_options
    if tuning_result_file_name:
        tuning_result_file = Path(tuning_result_output_dir) / tuning_result_file_name
        inference_settings["tuning_result_file"] = str(tuning_result_file)

        # Only set the tuning_op_result for ROCM. In this case, the tuning_result_file is not None.
        if tuning_op_result:
            inference_settings["tuning_op_result"] = tuning_op_result


def parse_tuning_result(*tuning_results):
    return min(tuning_results, key=lambda x: x["latency_ms"])


def get_thread_affinity_nums(affinity_str):
    affinities = affinity_str.split(";")
    return len(affinities)


class OrtPerfTuning(Pass):
    """Optimize ONNX Runtime inference settings."""

    _requires_user_script = True
    run_on_target = True

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        device = accelerator_spec.accelerator_type
        execution_provider = accelerator_spec.execution_provider

        return {
            "data_dir": PassConfigParam(
                type_=OLIVE_RESOURCE_ANNOTATIONS,
                category=ParamCategory.DATA,
                description="Directory of sample inference data.",
            ),
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                category=ParamCategory.OBJECT,
                description="Dataloader function to load data from given data_dir with given batch size.",
            ),
            "dataloader_func_kwargs": PassConfigParam(
                type_=Dict[str, Any],
                description="Keyword arguments for dataloader_func.",
            ),
            "batch_size": PassConfigParam(type_=int, description="Batch size for inference."),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                description="Data config to load data for computing latency.",
            ),
            "input_names": PassConfigParam(
                type_=list, default_value=None, description="Input names list for ONNX model."
            ),
            "input_shapes": PassConfigParam(
                type_=list, default_value=None, description="Input shapes list for ONNX model."
            ),
            "input_types": PassConfigParam(
                type_=list, default_value=None, description="Input types list for ONNX model."
            ),
            "device": PassConfigParam(
                type_=str, default_value=device, description="Device selected for tuning process."
            ),
            "cpu_cores": PassConfigParam(
                type_=int, default_value=None, description="CPU cores used for thread tuning."
            ),
            "io_bind": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether enable IOBinding Search for ONNX Runtime inference.",
            ),
            "enable_cuda_graph": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether enable CUDA Graph for CUDA execution provider.",
            ),
            "providers_list": PassConfigParam(
                type_=list,
                default_value=[execution_provider],
                description="Execution providers framework list to execute the ONNX models.",
            ),
            "execution_mode_list": PassConfigParam(
                type_=list, default_value=None, description="Parallelism list between operators."
            ),
            "opt_level_list": PassConfigParam(
                type_=list, default_value=None, description="Optimization level list for ONNX model."
            ),
            "trt_fp16_enable": PassConfigParam(
                type_=bool, default_value=False, description="Whether enable FP16 mode for TensorRT execution provider."
            ),
            "intra_thread_num_list": PassConfigParam(
                type_=list, default_value=[None], description="List of intra thread number for test."
            ),
            "inter_thread_num_list": PassConfigParam(
                type_=list, default_value=[None], description="List of inter thread number for test."
            ),
            "extra_session_config": PassConfigParam(
                type_=Dict[str, Any],
                default_value=None,
                description="Extra customized session options during tuning process.",
            ),
            "force_evaluate_other_eps": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether force to evaluate all execution providers"
                    " which are different with the associated execution provider."
                ),
            ),
            "enable_profiling": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether enable profiling for ONNX Runtime inference.",
            ),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        config = self._config_class(**config)
        # TODO(jambayk): decide on whether to ignore the output_model_path
        # if we want to ignore it, we can just return the model
        # otherwise save or symlink the original model to the output_model_path
        runner = PerfTuningRunner(self.accelerator_spec, config, data_root)
        return runner.tune_onnx_model(model)


class PerfTuningRunner:
    def __init__(self, accelerator_spec: AcceleratorSpec, config: Dict[str, Any], data_root: str = None):
        assert accelerator_spec, "accelerator_spec should not be None"
        assert config, "config should not be None"

        self.accelerator_spec = accelerator_spec
        self.config = config
        self.data_root = data_root

    def tune_onnx_model(self, model):
        latency_user_config = {}
        # which should be the same as the config in the metric
        config_dict = self.config.dict()

        # data_dir/dataloader_func will be passed to the metric as perf_tuning will leverage
        # the latency metric to run tune
        for eval_config in get_user_config_properties_from_metric_type(MetricType.LATENCY):
            if eval_config in config_dict:
                latency_user_config[eval_config] = config_dict.get(eval_config)
        if config_dict.get("dataloader_func_kwargs"):
            latency_user_config["func_kwargs"] = {"dataloader_func": config_dict.get("dataloader_func_kwargs")}
        latency_sub_types = [{"name": LatencySubType.AVG}]
        latency_metric_config = {
            "name": "latency",
            "type": MetricType.LATENCY,
            "sub_types": latency_sub_types,
            "user_config": latency_user_config,
            "data_config": config_dict.get("data_config"),
        }
        latency_metric = Metric(**latency_metric_config)

        # TODO(myguo): from the time being, the baseline evaluation doesn't enable enable_cuda_graph.
        # do we need enable it?
        io_bind = self.config.io_bind
        pretuning_inference_result = self.get_benchmark(model, latency_metric, io_bind=io_bind, tuning_result=None)

        tuning_op_result = pretuning_inference_result.get("tuning_op_result")
        tuning_results = []
        for provider, execution_mode, opt_level in generate_tuning_combos(self.config):
            provider, options = populate_provider_options(provider, self.config)  # noqa: PLW2901
            if provider == "CUDAExecutionProvider":
                # if enable_cuda_graph is True but the io_bind is False, the following errors will be raised.
                # onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : CUDA failure 700:
                #    an illegal memory access was encountered ; GPU=0 ; hostname=c93f1847c000000 ;
                #    file=/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_graph.cc ; line=49 ;
                #    expr=cudaGraphLaunch(graph_exec_, stream_);
                # [E:onnxruntime:Default, cuda_call.cc:116 CudaCall] CUDA failure 700:
                #    an illegal memory access was encountered ; GPU=0 ; hostname=c93f1847c000000 ;
                #    file=/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_execution_provider.cc ; line=286 ;
                #    expr=cudaStreamDestroy(stream_);
                # [E:onnxruntime:Default, cuda_call.cc:116 CudaCall] CUDNN failure 4:
                #    CUDNN_STATUS_INTERNAL_ERROR ; GPU=0 ; hostname=c93f1847c000000 ;
                #    file=/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_execution_provider.cc ; line=181 ;
                #    expr=cudnnDestroy(cudnn_handle_);
                io_bind = True
            elif provider == "CPUExecutionProvider":
                io_bind = False

            tuning_combo = (([provider], [options]), execution_mode, opt_level, io_bind)

            # TODO(myguo): we need disable the following check when we enable cache in perf tuning.
            if provider != self.accelerator_spec.execution_provider and not self.config.force_evaluate_other_eps:
                logger.warning(
                    "Ignore perf tuning for EP %s since current pass EP is %s",
                    provider,
                    self.accelerator_spec.execution_provider,
                )
                continue
            tuning_item = ["provider", "execution_mode", "ort_opt_level", "io_bind"]
            if not valid_config(tuning_combo, self.config):
                continue
            logger.info("Run tuning for: %s", list(zip(tuning_item, tuning_combo)))
            tuning_results.extend(self.threads_num_tuning(model, latency_metric, *tuning_combo, tuning_op_result))

        for tuning_result in tuning_results:
            logger.debug("Tuning result for %s: %s", tuning_result["test_name"], tuning_result["latency_ms"])

        best_result = parse_tuning_result(*tuning_results, pretuning_inference_result)
        logger.info("Best result: %s", best_result)
        # Both baseline and pertuning result should have the execution provider in the test_results.
        assert "execution_provider" in best_result, "execution_provider should be in best_result"
        optimized_model = copy.copy(model)
        optimized_model.inference_settings = {
            "execution_provider": best_result["execution_provider"],
            "provider_options": best_result["provider_options"],
            "io_bind": best_result["io_bind"],
            "tuning_op_result": best_result.get("tuning_op_result"),
        }
        session_options = best_result.get("session_options")
        if session_options is not None:
            optimized_model.inference_settings["session_options"] = session_options

        return optimized_model

    def get_benchmark(
        self,
        model,
        latency_metric,
        test_params=None,
        io_bind=False,
        tuning_result=None,
    ):
        import onnxruntime as ort

        from olive.evaluator.olive_evaluator import OliveEvaluatorFactory

        # prepare the inference_settings for metrics.
        tuning_result_file = None
        if test_params:
            assert "provider_options" in test_params, "provider_options should be in test_params"
            inference_settings = test_params
        else:
            inference_settings = copy.deepcopy(model.inference_settings) if model.inference_settings else {}
            # put the execution_provider and provider_options in inference_settings for baseline evaluation
            available_eps = ort.get_available_providers()
            execution_providers, provider_options = check_and_normalize_provider_args(
                self.config.providers_list, None, available_eps
            )
            inference_settings["execution_provider"] = execution_providers
            inference_settings["provider_options"] = provider_options

        if self.config.enable_profiling:
            if "session_options" not in inference_settings:
                inference_settings["session_options"] = {}
            inference_settings["session_options"]["enable_profiling"] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            enable_rocm_op_tuning(inference_settings, tuning_result, temp_dir)
            # set the session_options for metrics so that the evaluate will use them by default
            latency_metric.user_config.io_bind = io_bind
            latency_metric.user_config.inference_settings = {"onnx": inference_settings}

            session_name = generate_test_name(test_params, io_bind)
            logger.debug("Run benchmark for: %s", session_name)
            joint_key = joint_metric_key(latency_metric.name, latency_metric.sub_types[0].name)

            start_time = time.perf_counter()
            evaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
            metric_result = evaluator.evaluate(model, self.data_root, [latency_metric], self.config.device, None)

            end_time = time.perf_counter()
            latency_ms = metric_result[joint_key].value
            logger.debug("It takes %.5f seconds to benchmark for: %s", end_time - start_time, session_name)

            session_options = inference_settings.get("session_options")

            tuning_op_result = None
            tuning_result_file = inference_settings.get("tuning_result_file")
            if tuning_result_file:
                with Path(tuning_result_file).open() as f:
                    tuning_op_result = json.load(f)

            return {
                "test_name": session_name,
                "io_bind": io_bind,
                "latency_ms": latency_ms,
                "execution_provider": inference_settings["execution_provider"],
                "provider_options": inference_settings["provider_options"],
                "session_options": session_options if session_options else {},
                "tuning_op_result": tuning_op_result,
            }

    def threads_num_tuning(
        self,
        model,
        latency_metric,
        providers,
        execution_mode,
        ort_opt_level,
        io_bind,
        tuning_op_result,
    ):
        tuning_results = []
        provider, options = providers

        test_params = {
            "execution_provider": provider,
            "provider_options": options,
            "session_options": {
                "execution_mode": execution_mode,
                "graph_optimization_level": ort_opt_level,
            },
        }

        if self.config.extra_session_config:
            test_params["session_options"]["extra_session_config"] = self.config.extra_session_config

        try:
            for inter in self.config.inter_thread_num_list:
                if inter is not None:
                    test_params["session_options"]["inter_op_num_threads"] = inter
                for intra in self.config.intra_thread_num_list:
                    if intra is not None:
                        test_params["session_options"]["intra_op_num_threads"] = intra
                    tuning_result = self.threads_num_binary_search(
                        model,
                        latency_metric,
                        test_params,
                        io_bind,
                        tuning_op_result,
                    )
                    tuning_results.extend(tuning_result)
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception:
            logger.exception(
                "Optimization failed for tuning combo %s",
                (providers, execution_mode, ort_opt_level, io_bind),
            )

        return tuning_results

    def threads_num_binary_search(self, model, latency_metric, test_params, io_bind, tuning_op_result):
        """Binary search based benchmark for inter_op_num_threads and intra_op_num_threads."""
        import onnxruntime as ort
        import psutil

        # prepare the inter_op_num_threads/intra_op_num_threads to be tune.
        extra_session_config = test_params["session_options"].get("extra_session_config")
        if extra_session_config:
            affinity_str = extra_session_config.get("session.intra_op_thread_affinities")
            if affinity_str:
                test_params["session_options"]["intra_op_num_threads"] = get_thread_affinity_nums(affinity_str) + 1
                threads_names = ["inter_op_num_threads"]
        elif test_params["session_options"].get("execution_mode") == ort.ExecutionMode.ORT_SEQUENTIAL:
            threads_names = ["intra_op_num_threads"]
        else:
            threads_names = ["inter_op_num_threads", "intra_op_num_threads"]

        if (
            test_params["session_options"].get("inter_op_num_threads") is not None
            and test_params["session_options"].get("intra_op_num_threads") is not None
        ):
            # If user specify both inter_op_num_threads and intra_op_num_threads, we will not do tuning
            test_result = self.get_benchmark(model, latency_metric, test_params, io_bind, tuning_op_result)
            return [test_result]

        tuning_results = []

        def benchmark_with_threads_num(threads_name, threads_num):
            test_params["session_options"][threads_name] = threads_num
            test_result = self.get_benchmark(model, latency_metric, test_params, io_bind, tuning_op_result)
            tuning_results.append(test_result)
            return test_result["latency_ms"]

        for threads_name in threads_names:
            # set the upper bound and lower bound for binary search
            thread_num = test_params["session_options"].get(threads_name)
            if thread_num is not None:
                upper_threads_num = thread_num
                lower_threads_num = thread_num
            else:
                upper_threads_num = self.config.cpu_cores or psutil.cpu_count(logical=False)
                lower_threads_num = 1

            current_threads_num = lower_threads_num
            best_latency = None
            best_threads_num = None

            while lower_threads_num < upper_threads_num:
                benchmark_latency = benchmark_with_threads_num(threads_name, current_threads_num)

                if best_latency is None:
                    # the first time run benchmark, then change next to the upper bound
                    best_latency = benchmark_latency
                    best_threads_num = current_threads_num
                    current_threads_num = upper_threads_num
                elif best_latency < benchmark_latency:
                    mid_threads_num = lower_threads_num + (upper_threads_num - lower_threads_num) // 2
                    # the current benchmark is worse than best benchmark result.
                    # Just keep the best_latency and best_threads_num
                    if best_threads_num < current_threads_num:
                        # update the upper bound to middle if best benchmark is in lower side.
                        upper_threads_num = mid_threads_num
                        next_thread_num = upper_threads_num
                    else:
                        # update the lower bound to middle if best benchmark is in upper side.
                        # The benchmark result is worse than best benchmark and
                        # the thread num last time used is larger than current
                        lower_threads_num = mid_threads_num + 1
                        next_thread_num = lower_threads_num

                    current_threads_num = next_thread_num
                else:
                    mid_threads_num = lower_threads_num + (upper_threads_num - lower_threads_num) // 2

                    # the current benchmark result is better than best benchmark result
                    if best_threads_num < current_threads_num:
                        # If the thread number is in lower side, update the lower bound to middle
                        lower_threads_num = mid_threads_num + 1
                        next_thread_num = lower_threads_num
                    else:
                        # If the thread number is in upper side, update the upper bound to middle
                        upper_threads_num = mid_threads_num
                        next_thread_num = upper_threads_num

                    # Update the best_latency and best_threads_num for next comparison
                    best_latency = benchmark_latency
                    best_threads_num = current_threads_num
                    current_threads_num = next_thread_num

            # Pin the best threads num for inter_op_num_threads/intra_op_num_threads for next tuning config
            test_params["session_options"][threads_name] = best_threads_num

        return tuning_results
