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

    if optimization_config.model_analyzer_config:
        result_pbtxt_path = os.path.join(optimization_config.result_path, "olive_result.pbtxt")
        generate_ma_result(result_json_path, result_pbtxt_path, optimization_config.model_analyzer_config)

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

def generate_ma_result(json_file_path, result_pbtxt_path, ma_config_path):
    from google.protobuf import text_format, json_format
    from tritonclient.grpc import model_config_pb2

    with open(json_file_path) as json_file:
        olive_result = json.load(json_file)
        results = olive_result.get("all_tuning_results")
        best_test_name = olive_result.get("best_test_name")
        for result in results:
            if result.get("test_name") == best_test_name:
                execution_provider = result.get("execution_provider")
                env_vars = result.get("env_vars")
                session_options = result.get("session_options")
                break

    optimization_config = None
    sess_opt_parameters = None

    if best_test_name == "pretuning":
        optimization_config = {"graph": {"level": 1}}
    else:
        intra_op_thread_count = session_options.get("intra_op_num_threads")
        inter_op_thread_count = session_options.get("inter_op_num_threads")
        execution_mode = session_options.get("execution_mode")
        graph_optimization_level = session_options.get("graph_optimization_level")

        if graph_optimization_level in ["0", "1"]:
            opt_level = -1
        else: 
            opt_level = 1

        if execution_provider == "TensorrtExecutionProvider":
            tensorrt_accelerator = {"name": "tensorrt"}
            if env_vars.get("ORT_TENSORRT_FP16_ENABLE") == "1":
                tensorrt_accelerator["parameters"] = {"precision_mode": "FP16"}
            optimization_config = {
                "executionAccelerators": {"gpuExecutionAccelerator": [tensorrt_accelerator]},
                "graph": {"level": opt_level}
            }
        elif execution_provider == "OpenVINOExecutionProvider":
            optimization_config = {
                "executionAccelerators": {"cpuExecutionAccelerator": [{"name": "openvino"}]},
                "graph": {"level": opt_level}}
        else:
            optimization_config = {"graph": {"level": opt_level}}

        sess_opt_parameters = {}
        if intra_op_thread_count != "None":
            sess_opt_parameters["intra_op_thread_count"] = {"stringValue": intra_op_thread_count}
        if inter_op_thread_count != "None":
            sess_opt_parameters["inter_op_thread_count"] = {"stringValue": inter_op_thread_count}
        if execution_mode:
            execution_mode_flag = "0" if execution_mode == "ExecutionMode.ORT_SEQUENTIAL" else "1"
            sess_opt_parameters["execution_mode"] = {"stringValue": execution_mode_flag}

    with open(ma_config_path, 'r+') as f:
        config_str = f.read()
        protobuf_message = text_format.Parse(config_str, model_config_pb2.ModelConfig())
        model_dict = json_format.MessageToDict(protobuf_message)

    model_dict.update({"optimization": optimization_config})
    if sess_opt_parameters:
        model_dict.update({"parameters": sess_opt_parameters})


    protobuf_message = json_format.ParseDict(model_dict, model_config_pb2.ModelConfig())
    model_config_bytes = text_format.MessageToBytes(protobuf_message)

    with open(result_pbtxt_path, "wb") as f:
        f.write(model_config_bytes)