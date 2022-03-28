import json
import multiprocessing
import os

import onnxruntime as ort
import psutil

from .optimization.optimize_quantization import quantization_optimize
from .optimization.optimize_transformer import transformer_optimize
from .optimization.tuning_process import tune_onnx_model, get_benchmark
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optimize(optimization_config):
    if "OpenVINOExecutionProvider" in ort.get_available_providers() or "TensorrtExecutionProvider" in ort.get_available_providers():
        ort_deps_path = os.path.join(ort.__path__[0], "capi")
        os.environ["LD_LIBRARY_PATH"] = "{}:{}".format(os.environ.get("LD_LIBRARY_PATH"), ort_deps_path)

    multiprocessing.set_start_method("spawn", force=True)

    pretuning_inference_result = get_benchmark(optimization_config)
    if not optimization_config.throughput_tuning_enabled:
        optimization_config.pretuning_latency_ms = pretuning_inference_result["latency_ms"]["avg"]

    if optimization_config.transformer_enabled:
        transformer_optimize(optimization_config)

    if optimization_config.quantization_enabled:
        quantization_optimize(optimization_config)

    tuning_results = tune_onnx_model(optimization_config)

    olive_result = parse_tuning_result(optimization_config, *tuning_results, pretuning_inference_result)

    result_json_path = os.path.join(optimization_config.result_path, "olive_result.json")

    with open(result_json_path, 'w') as f:
        json.dump(olive_result, f, indent=4)

    if optimization_config.throughput_tuning_enabled:
        for file_name in os.listdir(optimization_config.result_path):
            if file_name.startswith("mlperf"):
                os.remove(os.path.join(optimization_config.result_path, file_name))

    logger.info("Optimization succeeded, OLive tuning result written in {}".format(result_json_path))

    return olive_result


def parse_tuning_result(optimization_config, *tuning_results):
    if optimization_config.throughput_tuning_enabled:
        best_test_name = max(tuning_results, key=lambda x: x["throughput"])["test_name"]
    else:
        best_test_name = min(tuning_results, key=lambda x: x["latency_ms"]["avg"])["test_name"]

    successful_eps = list(set([result.get("execution_provider") for result in tuning_results
                          if result.get("execution_provider")]))

    olive_result = dict()
    olive_result["model_path"] = optimization_config.model_path
    olive_result["tuning_ort_version"] = ort.__version__
    olive_result["core_num"] = psutil.cpu_count(logical=False)
    olive_result["quantization_enabled"] = str(optimization_config.quantization_enabled)
    olive_result["transformer_enabled"] = str(optimization_config.transformer_enabled)
    olive_result["transformer_args"] = optimization_config.transformer_args
    olive_result["pretuning_test_name"] = "pretuning"
    olive_result["tested_eps"] = optimization_config.providers_list
    olive_result["successful_eps"] = successful_eps
    olive_result["best_test_name"] = best_test_name
    olive_result["total_tests_num"] = len(tuning_results)
    olive_result["all_tuning_results"] = tuning_results

    return olive_result
