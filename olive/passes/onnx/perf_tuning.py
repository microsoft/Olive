# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import itertools
import logging
from typing import Any, Callable, Dict, Union

from olive.data.config import DataConfig
from olive.evaluator.metric import LatencySubType, Metric, MetricType, joint_metric_key
from olive.evaluator.metric_config import get_user_config_properties_from_metric_type
from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


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

    if config.io_bind:
        io_bind_list = [True, False]
    else:
        io_bind_list = [False]

    tuning_combos = itertools.product(providers_list, execution_mode_list, opt_level_list, io_bind_list)
    yield from tuning_combos


def valid_config(tuning_combos, config):
    # the order of combos: "provider", "execution_mode", "ort_opt_level", "io_bind"

    # Parallel execution mode does not support the CUDA Execution Provider.
    # So ORT will make the execution mode sequential when it uses the CUDA Execution Provider.

    # if the first combo is CPUExecutionProvider, then the io_bind should not be True
    if tuning_combos[0] == "CPUExecutionProvider" and tuning_combos[3]:
        logger.info("[Ignored] Because EP is CPUExecutionProvider, the io_bind should not be True")
        return False
    if tuning_combos[0] != "CUDAExecutionProvider" and config.enable_cuda_graph:
        logger.info("[Ignored] Because EP is not CUDAExecutionProvider, the enable_cuda_graph is ignored")
        return True
    if tuning_combos[0] != "TensorrtExecutionProvider" and config.trt_fp16_enable:
        logger.info("[Ignored] Because EP is not TensorrtExecutionProvider, the trt_fp16_enable is ignored")
        return True
    return True


def tune_onnx_model(model, data_root, config):
    latency_user_config = {}
    # which should be the same as the config in the metric
    config_dict = config.dict()

    # data_dir/dataloader_func will be passed to the metric as perf_tuning will leverage
    # the latency metric to run tune
    for eval_config in get_user_config_properties_from_metric_type(MetricType.LATENCY):
        if eval_config in config_dict:
            latency_user_config[eval_config] = config_dict.get(eval_config)
    latency_sub_types = [{"name": LatencySubType.AVG}]
    latency_metric_config = {
        "name": "latency",
        "type": MetricType.LATENCY,
        "sub_types": latency_sub_types,
        "user_config": latency_user_config,
        "data_config": config_dict.get("data_config"),
    }
    latency_metric = Metric(**latency_metric_config)

    pretuning_inference_result = get_benchmark(model, data_root, latency_metric, config)

    tuning_results = []
    for tuning_combo in generate_tuning_combos(config):
        tuning_item = ["provider", "execution_mode", "ort_opt_level", "io_bind"]
        logger.info("Run tuning for: {}".format(list(zip(tuning_item, tuning_combo))))
        if not valid_config(tuning_combo, config):
            continue
        tuning_results.extend(threads_num_tuning(model, data_root, latency_metric, config, tuning_combo))

    for tuning_result in tuning_results:
        logger.debug("Tuning result: {}".format(tuning_result["latency_ms"]))

    best_result = parse_tuning_result(*tuning_results, pretuning_inference_result)
    logger.info("Best result: {}".format(best_result))
    if best_result.get("test_name") != "pretuning":
        optimized_model = copy.copy(model)
        optimized_model.inference_settings = {
            "execution_provider": best_result.get("execution_provider"),
            "session_options": best_result.get("session_options"),
        }

        return optimized_model
    else:
        return model


def threads_num_tuning(model, data_root, latency_metric, config, tuning_combo):
    tuning_results = []
    provider = tuning_combo[0]
    execution_mode = tuning_combo[1]
    ort_opt_level = tuning_combo[2]
    io_bind = tuning_combo[3]

    test_params = dict()

    # params starts with _ are not used in inference setting, we need add special handling for io_bind
    test_params["_io_bind"] = io_bind

    if provider == "TensorrtExecutionProvider":
        test_params["execution_provider"] = [
            (
                "TensorrtExecutionProvider",
                {"trt_fp16_enable": config.trt_fp16_enable},
            )
        ]
    elif provider == "CUDAExecutionProvider":
        test_params["execution_provider"] = [
            (
                "CUDAExecutionProvider",
                {"enable_cuda_graph": config.enable_cuda_graph},
            )
        ]
        if config.enable_cuda_graph:
            test_params["_io_bind"] = True
    else:
        test_params["execution_provider"] = [(provider, dict())]
    test_params["session_options"] = {
        "execution_mode": execution_mode,
        "graph_optimization_level": ort_opt_level,
        "extra_session_config": config.extra_session_config,
    }

    try:
        for inter in config.inter_thread_num_list:
            test_params["session_options"]["inter_op_num_threads"] = inter
            for intra in config.intra_thread_num_list:
                test_params["session_options"]["intra_op_num_threads"] = intra
                threads_num_binary_search(model, data_root, latency_metric, config, test_params, tuning_results)
    except Exception:
        logger.error("Optimization failed for tuning combo {}".format(tuning_combo), exc_info=True)
        pass

    return tuning_results


def threads_num_binary_search(model, data_root, latency_metric, config, test_params, tuning_results):
    import onnxruntime as ort
    import psutil

    if test_params["session_options"].get("extra_session_config"):
        extra_session_config = test_params["session_options"].get("extra_session_config")
        if extra_session_config.get("session.intra_op_thread_affinities"):
            affinity_str = extra_session_config.get("session.intra_op_thread_affinities")
            test_params["session_options"]["intra_op_num_threads"] = get_thread_affinity_nums(affinity_str) + 1
            threads_names = ["inter_op_num_threads"]
    elif test_params["session_options"].get("execution_mode") == ort.ExecutionMode.ORT_SEQUENTIAL:
        threads_names = ["intra_op_num_threads"]
    else:
        threads_names = ["inter_op_num_threads", "intra_op_num_threads"]
    best_latency = None

    if test_params["session_options"].get("inter_op_num_threads") and test_params["session_options"].get(
        "intra_op_num_threads"
    ):
        test_result = get_benchmark(model, data_root, latency_metric, config, test_params)
        tuning_results.append(test_result)
        best_latency = test_result["latency_ms"]
    else:
        for threads_name in threads_names:
            thread_num = test_params["session_options"].get(threads_name)
            if thread_num:
                upper_threads_num = thread_num
                lower_threads_num = thread_num
            else:
                upper_threads_num = config.cpu_cores or psutil.cpu_count(logical=False)
                lower_threads_num = 1

            best_threads_num = lower_threads_num
            current_threads_num = lower_threads_num
            test_params["session_options"][threads_name] = current_threads_num

            test_result = get_benchmark(model, data_root, latency_metric, config, test_params)
            tuning_results.append(test_result)

            best_latency = test_result["latency_ms"]

            current_threads_num = upper_threads_num

            while lower_threads_num < upper_threads_num:
                test_params["session_options"][threads_name] = current_threads_num

                test_result = get_benchmark(model, data_root, latency_metric, config, test_params)
                tuning_results.append(test_result)

                mid_threads_num = lower_threads_num + (upper_threads_num - lower_threads_num) // 2
                if best_latency and best_latency < test_result["latency_ms"]:
                    upper_threads_num = mid_threads_num
                    current_threads_num = upper_threads_num
                else:
                    lower_threads_num = mid_threads_num + 1
                    best_threads_num = current_threads_num
                    current_threads_num = lower_threads_num

            test_params["session_options"][threads_name] = best_threads_num


def generate_test_name(test_params):
    if test_params:
        test_name = "_".join(["_".join([str(v) for v in i]) for i in test_params.items()])
    else:
        test_name = "pretuning"
    return test_name


def get_benchmark(model, data_root, latency_metric, config, test_params=None):
    from olive.evaluator.olive_evaluator import OliveEvaluatorFactory

    test_result = {}
    session_name = generate_test_name(test_params)
    test_result["test_name"] = session_name

    if test_params:
        # params starts with _ are not used in inference setting, we need add special handling for io_bind
        io_bind = test_params.pop("_io_bind", False)
        latency_metric.user_config.io_bind = io_bind

        latency_metric.user_config.inference_settings = {"onnx": test_params}
        test_result["execution_provider"] = test_params.get("execution_provider")
        test_result["session_options"] = test_params.get("session_options").copy()
        test_result["io_bind"] = io_bind

        # add the io_bind back to test_params
        test_params["_io_bind"] = io_bind
    evaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
    joint_key = joint_metric_key(latency_metric.name, latency_metric.sub_types[0].name)
    test_result["latency_ms"] = evaluator.evaluate(
        model, data_root, [latency_metric], config.device, config.providers_list
    )[joint_key].value
    return test_result


def parse_tuning_result(*tuning_results):
    best_result = min(tuning_results, key=lambda x: x["latency_ms"])
    return best_result


def get_thread_affinity_nums(affinity_str):
    affinities = affinity_str.split(";")
    return len(affinities)


class OrtPerfTuning(Pass):
    """Optimize ONNX Runtime inference settings."""

    _requires_user_script = True

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the
        accelerator spec information.
        """
        return False

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
                type_=str, default_value="cpu", description="Device selected for tuning process."
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
                default_value=None,
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
        }

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        # TODO remove this when we have a concrete investigation on the backcompat issue
        if not config.get("providers_list"):
            # add the provider to the config if user doesn't provide the execution providers
            config["providers_list"] = [self.accelerator_spec.execution_provider]

        if not config.get("device"):
            config["device"] = self.accelerator_spec.accelerator_type

        config = self._config_class(**config)
        # TODO: decide on whether to ignore the output_model_path
        # if we want to ignore it, we can just return the model
        # otherwise save or symlink the original model to the output_model_path
        return tune_onnx_model(model, data_root, config)
