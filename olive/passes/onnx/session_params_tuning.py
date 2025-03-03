# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Type, Union

import onnxruntime as ort

from olive.common.config_utils import validate_config
from olive.common.ort_inference import check_and_normalize_provider_args
from olive.common.pydantic_v1 import Extra
from olive.data.config import DataConfig
from olive.evaluator.metric import LatencySubType, Metric, MetricType
from olive.evaluator.metric_result import joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.exception import EXCEPTIONS_TO_RAISE
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config
from olive.search.search_parameter import Categorical

logger = logging.getLogger(__name__)


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


def get_thread_affinity_nums(affinity_str):
    affinities = affinity_str.split(";")
    return len(affinities)


class OrtSessionParamsTuning(Pass):
    """Optimize ONNX Runtime inference settings."""

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        device = accelerator_spec.accelerator_type
        execution_provider = accelerator_spec.execution_provider

        return {
            **get_user_script_data_config(),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                description="Data config to load data for computing latency.",
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
                type_=str,
                default_value=execution_provider,
                search_defaults=Categorical(AcceleratorLookup.get_execution_providers_for_device(device)),
                description="Execution providers framework list to execute the ONNX models.",
            ),
            "provider_options_list": PassConfigParam(
                type_=Dict[str, Any],
                default_value={},
                search_defaults=Categorical([{}]),
                description="Execution provider options to execute the ONNX models.",
            ),
            "execution_mode_list": PassConfigParam(
                type_=int,
                default_value=None,
                search_defaults=Categorical([None]),
                description="Parallelism list between operators.",
            ),
            "opt_level_list": PassConfigParam(
                type_=int,
                default_value=None,
                search_defaults=Categorical([None]),
                description="Optimization level list for ONNX model.",
            ),
            "trt_fp16_enable": PassConfigParam(
                type_=bool, default_value=False, description="Whether enable FP16 mode for TensorRT execution provider."
            ),
            "intra_thread_num_list": PassConfigParam(
                type_=int,
                default_value=None,
                search_defaults=Categorical([None]),
                description="List of intra thread number for test.",
            ),
            "inter_thread_num_list": PassConfigParam(
                type_=int,
                default_value=None,
                search_defaults=Categorical([None]),
                description="List of inter thread number for test.",
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

    @classmethod
    def validate_config(
        cls,
        config: Type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        """Validate the search point for the pass."""
        if not super().validate_config(config, accelerator_spec):
            return False

        # Rename the search parameters with atomic/singular names for clarity
        config.__class__.__config__.extra = Extra.allow
        config.execution_provider = config.providers_list
        config.provider_options = config.provider_options_list
        config.execution_mode = config.execution_mode_list
        config.opt_level = config.opt_level_list
        config.intra_op_thread_count = config.intra_thread_num_list
        config.inter_op_thread_count = config.inter_thread_num_list

        if config.execution_provider == "CUDAExecutionProvider" and not config.io_bind:
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
            return False
        elif config.execution_provider == "CPUExecutionProvider" and config.io_bind:
            # if the first combo is CPUExecutionProvider, then the io_bind should not be True
            logger.info("[Ignored] Because EP is CPUExecutionProvider, the io_bind should not be True")
            return False

        # Parallel execution mode does not support the CUDA Execution Provider.
        # So ORT will make the execution mode sequential when it uses the CUDA Execution Provider.

        if config.execution_provider != "CUDAExecutionProvider" and config.enable_cuda_graph:
            logger.info("[Ignored] Because EP is not CUDAExecutionProvider, the enable_cuda_graph is ignored")
            return True
        if config.execution_provider != "TensorrtExecutionProvider" and config.trt_fp16_enable:
            logger.info("[Ignored] Because EP is not TensorrtExecutionProvider, the trt_fp16_enable is ignored")
            return True
        if (
            config.execution_provider == "CUDAExecutionProvider"
            and config.enable_cuda_graph
            and config.execution_mode == ort.ExecutionMode.ORT_PARALLEL.value
        ):
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

        # TODO(myguo): we need disable the following check when we enable cache in perf tuning.
        if config.execution_provider != accelerator_spec.execution_provider and not config.force_evaluate_other_eps:
            logger.warning(
                "Ignore perf tuning for EP %s since current pass EP is %s",
                config.execution_provider,
                accelerator_spec.execution_provider,
            )
            return False

        return True

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        # Rename the search parameters with atomic/singular names for clarity
        config.__class__.__config__.extra = Extra.allow
        config = config.copy()
        config.execution_provider = config.providers_list
        config.provider_options = config.provider_options_list
        config.execution_mode = config.execution_mode_list
        config.opt_level = config.opt_level_list
        config.intra_op_thread_count = config.intra_thread_num_list
        config.inter_op_thread_count = config.inter_thread_num_list

        if config.execution_provider == "TensorrtExecutionProvider":
            config.provider_options["trt_fp16_enable"] = config.trt_fp16_enable
        elif config.execution_provider == "CUDAExecutionProvider":
            config.provider_options["enable_cuda_graph"] = config.enable_cuda_graph

        # TODO(jambayk): decide on whether to ignore the output_model_path
        # if we want to ignore it, we can just return the model
        # otherwise save or symlink the original model to the output_model_path
        if config.data_config:
            config.data_config = validate_config(config.data_config, DataConfig)

        latency_metric_config = {
            "name": "latency",
            "type": MetricType.LATENCY,
            "sub_types": [{"name": LatencySubType.AVG}],
            "data_config": config.data_config,
        }
        latency_metric = Metric(**latency_metric_config)

        provider, options = config.execution_provider, config.provider_options

        # TODO(myguo): For the time being, the baseline evaluation doesn't enable
        # enable_cuda_graph. Do we need enable it?
        try:
            pretuning_inference_result = self.evaluate(
                model, config, latency_metric, io_bind=config.io_bind, tuning_op_result=None
            )
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception:
            logger.exception("Baseline evaluation failed!")
            return copy.copy(model)

        tuning_op_result = pretuning_inference_result.get("tuning_op_result")

        tuning_params = {
            "provider": provider,
            "options": options,
            "execution_mode": config.execution_mode,
            "ort_opt_level": config.opt_level,
            "io_bind": config.io_bind,
        }

        logger.info("Running tuning with params: %s", tuning_params)
        tuning_result = self.get_benchmark(
            model, config, latency_metric, tuning_op_result=tuning_op_result, **tuning_params
        )
        logger.debug("Tuning result with params: %s = %s", tuning_params, tuning_result["latency_ms"])

        # Both baseline and pertuning result should have the execution provider in the test_results.
        assert "execution_provider" in tuning_result, "execution_provider should be in tuning_result"

        optimized_model = copy.copy(model)
        optimized_model.inference_settings = {
            "execution_provider": tuning_result["execution_provider"],
            "provider_options": tuning_result["provider_options"],
            "io_bind": tuning_result["io_bind"],
            "tuning_op_result": tuning_result.get("tuning_op_result"),
        }
        session_options = tuning_result.get("session_options")
        if session_options is not None:
            optimized_model.inference_settings["session_options"] = session_options

        return optimized_model

    def evaluate(
        self,
        model,
        config,
        latency_metric,
        test_params=None,
        io_bind=False,
        tuning_op_result=None,
    ):
        # prepare the inference_settings for metrics.
        if test_params:
            assert "provider_options" in test_params, "provider_options should be in test_params"
            inference_settings = test_params
        else:
            inference_settings = copy.deepcopy(model.inference_settings) if model.inference_settings else {}
            # put the execution_provider and provider_options in inference_settings for baseline evaluation
            available_eps = ort.get_available_providers()
            execution_providers, provider_options = check_and_normalize_provider_args(
                [config.execution_provider], None, available_eps
            )
            inference_settings["execution_provider"] = execution_providers
            inference_settings["provider_options"] = provider_options

        if config.enable_profiling:
            if "session_options" not in inference_settings:
                inference_settings["session_options"] = {}
            inference_settings["session_options"]["enable_profiling"] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            enable_rocm_op_tuning(inference_settings, tuning_op_result, temp_dir)
            # set the session_options for metrics so that the evaluate will use them by default
            latency_metric.user_config.io_bind = io_bind
            latency_metric.user_config.inference_settings = {"onnx": inference_settings}

            joint_key = joint_metric_key(latency_metric.name, latency_metric.sub_types[0].name)

            evaluator_config = OliveEvaluatorConfig(metrics=[latency_metric])
            evaluator = evaluator_config.create_evaluator(model)
            metric_result = evaluator.evaluate(model, evaluator_config.metrics, config.device, None)

            latency_ms = metric_result[joint_key].value
            session_options = inference_settings.get("session_options")

            tuning_op_result = None
            tuning_result_file = inference_settings.get("tuning_result_file")
            if tuning_result_file:
                with Path(tuning_result_file).open() as f:
                    tuning_op_result = json.load(f)

            return {
                "io_bind": io_bind,
                "latency_ms": latency_ms,
                "execution_provider": inference_settings["execution_provider"],
                "provider_options": inference_settings["provider_options"],
                "session_options": session_options if session_options else {},
                "tuning_op_result": tuning_op_result,
            }

    def get_benchmark(
        self,
        model,
        config,
        latency_metric,
        provider,
        options,
        execution_mode,
        ort_opt_level,
        io_bind,
        tuning_op_result,
    ):
        import psutil

        test_params = {
            "execution_provider": [provider],
            "provider_options": [options],
            "session_options": {
                "execution_mode": execution_mode,
                "graph_optimization_level": ort_opt_level,
            },
        }

        if config.extra_session_config:
            test_params["session_options"]["extra_session_config"] = config.extra_session_config

        if config.inter_op_thread_count is not None:
            test_params["session_options"]["inter_op_num_threads"] = config.inter_op_thread_count

        if config.intra_op_thread_count is not None:
            test_params["session_options"]["intra_op_num_threads"] = config.intra_op_thread_count

        # prepare the inter_op_num_threads/intra_op_num_threads to be tune.
        thread_names = []
        extra_session_config = test_params["session_options"].get("extra_session_config")
        if extra_session_config:
            affinity_str = extra_session_config.get("session.intra_op_thread_affinities")
            if affinity_str:
                test_params["session_options"]["intra_op_num_threads"] = get_thread_affinity_nums(affinity_str) + 1
                thread_names = ["inter_op_num_threads"]

        if not thread_names:
            if test_params["session_options"].get("execution_mode") == ort.ExecutionMode.ORT_SEQUENTIAL:
                thread_names = ["intra_op_num_threads"]
            else:
                thread_names = ["inter_op_num_threads", "intra_op_num_threads"]

        for thread_name in thread_names:
            # set the upper bound and lower bound for binary search
            thread_count = test_params["session_options"].get(thread_name)
            if thread_count is None:
                thread_count = config.cpu_cores or psutil.cpu_count(logical=False)

            test_params["session_options"][thread_name] = thread_count

        try:
            return self.evaluate(model, config, latency_metric, test_params, io_bind, tuning_op_result)
        except EXCEPTIONS_TO_RAISE:
            raise
        except Exception:
            logger.exception(
                "Optimization failed for tuning params %s",
                (provider, options, execution_mode, ort_opt_level, io_bind),
            )
            return None
