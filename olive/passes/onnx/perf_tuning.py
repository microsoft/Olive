# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import itertools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import onnxruntime as ort
import psutil

from olive.evaluator.evaluation import evaluate_latency
from olive.evaluator.metric import LatencySubType, Metric, MetricType
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


def generate_tuning_combos(config):
    providers_list = config.providers_list if config.providers_list else ort.get_available_providers()
    execution_mode_list = (
        config.execution_mode_list
        if config.execution_mode_list
        else [ort.ExecutionMode.ORT_SEQUENTIAL.value, ort.ExecutionMode.ORT_PARALLEL.value]
    )
    opt_level_list = config.opt_level_list if config.opt_level_list else [99]

    io_bind_list = None
    if isinstance(config.io_bind, list):
        io_bind_list = config.io_bind
    elif isinstance(config.io_bind, bool):
        io_bind_list = [config.io_bind]
    else:
        io_bind_list = [False]

    tuning_combos = itertools.product(providers_list, execution_mode_list, opt_level_list, io_bind_list)
    yield from tuning_combos


def tune_onnx_model(model, config):
    latency_user_config = config.dict()
    latency_user_config.pop("io_bind")
    latency_metric = Metric(
        name="latency", type=MetricType.LATENCY, sub_type=LatencySubType.AVG, user_config=latency_user_config
    )

    pretuning_inference_result = get_benchmark(model, latency_metric, config)

    tuning_results = []
    for tuning_combo in generate_tuning_combos(config):
        logger.info("Run tuning for: {}".format(tuning_combo))
        if tuning_combo[0] == "CPUExecutionProvider" and tuning_combo[3]:
            continue
        tuning_results.extend(threads_num_tuning(model, latency_metric, config, tuning_combo))
        logger.info("Current tuning results: {}".format(tuning_results[-1]["latency_ms"]))

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


def threads_num_tuning(model, latency_metric, config, tuning_combo):
    tuning_results = []
    provider = tuning_combo[0]
    execution_mode = tuning_combo[1]
    ort_opt_level = tuning_combo[2]
    io_bind = tuning_combo[3]

    test_params = dict()
    if provider == "TensorrtExecutionProvider":
        test_params["execution_provider"] = [
            (
                "TensorrtExecutionProvider",
                {"trt_fp16_enable": config.trt_fp16_enable},
            )
        ]
    else:
        test_params["execution_provider"] = [(provider, dict())]
    test_params["session_options"] = {
        "execution_mode": execution_mode,
        "graph_optimization_level": ort_opt_level,
        "extra_session_config": config.extra_session_config,
    }
    # params starts with _ are not used in inference setting, we need add special handling for io_bind
    test_params["_io_bind"] = io_bind
    try:
        for inter in config.inter_thread_num_list:
            test_params["session_options"]["inter_op_num_threads"] = inter
            for intra in config.intra_thread_num_list:
                test_params["session_options"]["intra_op_num_threads"] = intra
                threads_num_binary_search(model, latency_metric, config, test_params, tuning_results)
    except Exception:
        logging.error("Optimization failed for tuning combo {}".format(tuning_combo))
        pass

    return tuning_results


def threads_num_binary_search(model, latency_metric, config, test_params, tuning_results):
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
        test_result = get_benchmark(model, latency_metric, config, test_params)
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

            test_result = get_benchmark(model, latency_metric, config, test_params)
            tuning_results.append(test_result)

            best_latency = test_result["latency_ms"]

            current_threads_num = upper_threads_num

            while lower_threads_num < upper_threads_num:
                test_params["session_options"][threads_name] = current_threads_num

                test_result = get_benchmark(model, latency_metric, config, test_params)
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


def get_benchmark(model, latency_metric, config, test_params=None):
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
    test_result["latency_ms"] = evaluate_latency(model, latency_metric, config.device)
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
    def _default_config() -> Dict[str, Dict[str, Any]]:
        return {
            "data_dir": PassConfigParam(
                type_=Union[Path, str], is_path=True, description="Directory of sample inference data."
            ),
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                required=True,
                is_object=True,
                description="Dataloader function to load data from given data_dir with given batch size.",
            ),
            "batch_size": PassConfigParam(type_=int, required=True, description="Batch size for inference."),
            "device": PassConfigParam(type_=str, default="cpu", description="Device selected for tuning process."),
            "cpu_cores": PassConfigParam(type_=int, default=None, description="CPU cores used for thread tuning."),
            "io_bind": PassConfigParam(
                type_=Union[bool, List[bool]],
                default=False,
                description="Whether enable IOBingding for ONNX Runimte infernece.",
            ),
            "providers_list": PassConfigParam(
                type_=list, default=None, description="Execution providers framework list to execute the ONNX models."
            ),
            "execution_mode_list": PassConfigParam(
                type_=list, default=None, description="Parallelism list between operators."
            ),
            "opt_level_list": PassConfigParam(
                type_=list, default=None, description="Optimization level list for ONNX model."
            ),
            "trt_fp16_enable": PassConfigParam(
                type_=bool, default=False, description="Whether enable FP16 mode for TensorRT execution provider."
            ),
            "intra_thread_num_list": PassConfigParam(
                type_=list, default=[None], description="List of intra thread number for test."
            ),
            "inter_thread_num_list": PassConfigParam(
                type_=list, default=[None], description="List of inter thread number for test."
            ),
            "extra_session_config": PassConfigParam(
                type_=Dict[str, Any],
                default=None,
                description="Extra customized session options during tuning process.",
            ),
        }

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        config = self._config_class(**config)
        # TODO: decide on whether to ignore the output_model_path
        # if we want to ignore it, we can just return the model
        # otherwise save or symlink the original model to the output_model_path
        return tune_onnx_model(model, config)
