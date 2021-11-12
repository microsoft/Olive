import itertools
import logging
import os
import re
import time
from multiprocessing import Barrier, Process, Manager

import numpy as np
import onnxruntime as ort
import psutil

from .mlperf_dataset import Dataset
from .server_runner import ServerRunner
from ..constants import SUB_PROCESS_NAME_PREFIX

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cpu_cores = psutil.cpu_count(logical=False)

ort.set_default_logger_severity(3)


def tune_onnx_model(optimization_config):
    all_test_results = []
    for tuning_combo in generate_tuning_combos(optimization_config):
        all_test_results.extend(threads_num_tuning(optimization_config, tuning_combo))
    return all_test_results


def generate_tuning_combos(optimization_config):
    tuning_combos = itertools.product(optimization_config.omp_wait_policy_list, optimization_config.kmp_affinity,
                                      optimization_config.omp_max_active_levels, optimization_config.providers_list,
                                      optimization_config.execution_mode_list, optimization_config.ort_opt_level_list)
    yield from tuning_combos


def threads_num_tuning(optimization_config, tuning_combo):
    cpu_cores = psutil.cpu_count(logical=False)
    tuning_results = []

    os.environ["OMP_WAIT_POLICY"] = tuning_combo[0]
    os.environ["KMP_AFFINITY"] = tuning_combo[1]
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = tuning_combo[2]
    provider = tuning_combo[3]
    execution_mode = tuning_combo[4]
    ort_opt_level = tuning_combo[5]

    test_params = dict()
    test_params["execution_provider"] = provider
    test_params["execution_mode"] = execution_mode
    test_params["graph_optimization_level"] = ort_opt_level

    if provider == "TensorrtExecutionProvider" and optimization_config.trt_fp16_enabled:
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

    try:
        for inter in optimization_config.inter_thread_num_list:
            test_params["inter_op_num_threads"] = inter
            for intra in optimization_config.intra_thread_num_list:
                test_params["intra_op_num_threads"] = intra
                threads_num_binary_search(optimization_config, test_params, tuning_results, cpu_cores)

    except Exception:
        logger.error("Optimization failed for tuning combo {}".format(tuning_combo))
        pass

    return tuning_results


def threads_num_binary_search(optimization_config, test_params, tuning_results, cpu_cores):
    threads_names = ["inter_op_num_threads", "intra_op_num_threads"]
    best_throughput = None
    best_latency = None

    for threads_name in threads_names:
        thread_num = test_params.get(threads_name)
        if thread_num is not None:
            upper_threads_num = thread_num
            lower_threads_num = thread_num
        else:
            upper_threads_num = cpu_cores
            lower_threads_num = 1

        best_threads_num = lower_threads_num
        current_threads_num = lower_threads_num
        test_params[threads_name] = current_threads_num

        test_result = get_benchmark(optimization_config, test_params)
        if test_result:
            tuning_results.append(test_result)

        if optimization_config.throughput_tuning_enabled:
            best_throughput = test_result["throughput"]
        else:
            best_latency = test_result["latency_ms"]["avg"]

        current_threads_num = upper_threads_num

        while lower_threads_num < upper_threads_num:
            test_params[threads_name] = current_threads_num

            test_result = get_benchmark(optimization_config, test_params)
            if test_result:
                tuning_results.append(test_result)

            mid_threads_num = lower_threads_num + (upper_threads_num - lower_threads_num) // 2
            if (best_throughput and best_throughput > test_result["throughput"]) or (best_latency and best_latency < test_result["latency_ms"]["avg"]):
                upper_threads_num = mid_threads_num
                current_threads_num = upper_threads_num
            else:
                lower_threads_num = mid_threads_num + 1
                best_threads_num = current_threads_num
                current_threads_num = lower_threads_num

        test_params[threads_name] = best_threads_num


def generate_test_name(test_params):
    test_name = "_".join(["_".join([str(v) for v in i]) for i in test_params.items()])
    for env_var in ["OMP_WAIT_POLICY", "KMP_AFFINITY", "OMP_MAX_ACTIVE_LEVELS"]:
        env_val = os.getenv(env_var)
        if env_val:
            test_name += "_{}_{}".format(env_var, env_val)
    return test_name


def create_inference_session(model_path, test_params=None):
    sess_options = ort.SessionOptions()

    if test_params:
        session_name = generate_test_name(test_params)

        execution_provider = test_params.get("execution_provider")
        inter_op_num_threads = test_params.get("inter_op_num_threads")
        intra_op_num_threads = test_params.get("intra_op_num_threads")
        execution_mode = test_params.get("execution_mode")
        graph_optimization_level = test_params.get("graph_optimization_level")
        if inter_op_num_threads:
            sess_options.inter_op_num_threads = inter_op_num_threads
        if intra_op_num_threads:
            sess_options.intra_op_num_threads = intra_op_num_threads
            os.environ['OMP_NUM_THREADS'] = str(intra_op_num_threads)
        if execution_mode:
            sess_options.execution_mode = execution_mode
        if graph_optimization_level:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
        if execution_provider:
            onnx_session = ort.InferenceSession(model_path, sess_options, providers=[execution_provider])
        else:
            onnx_session = ort.InferenceSession(model_path, sess_options)

    else:
        session_name = "pretuning"
        execution_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        onnx_session = ort.InferenceSession(model_path, sess_options, providers=[execution_provider])

    return onnx_session, session_name


def get_benchmark(optimization_config, test_params=None):

    manager = Manager()
    test_result = manager.dict()
    if optimization_config.throughput_tuning_enabled:
        main_process = Process(name="main_process", target=get_throughput,
                               args=(optimization_config, test_params, test_result))
    else:
        main_process = Process(name="main_process", target=get_latency,
                               args=(optimization_config, test_params, test_result))

    process_list = []
    num_of_background = optimization_config.concurrency_num - 1
    if num_of_background > 0:
        synchronizer = Barrier(num_of_background + 1)
        for i in range(0, num_of_background):
            p = Process(name="{}_{}".format(SUB_PROCESS_NAME_PREFIX, i), target=concurrent_inference,
                        args=(synchronizer, optimization_config, test_params))
            p.start()
            process_list.append(p)
        synchronizer.wait()

    # execute main func, we only collect benchmark from this main func
    main_process.start()
    main_process.join()

    # once the main func finished, stop child process
    for p in process_list:
        if p.is_alive():
            p.terminate()
            logger.info("PID {} is killed".format(p.pid))

    return dict(test_result)


def concurrent_inference(synchronizer, optimization_config, test_params=None):
    logger.info("[{}] process start".format(os.getpid()))
    synchronizer.wait()

    # execute inference
    if optimization_config.throughput_tuning_enabled:
        get_throughput(optimization_config=optimization_config, test_params=test_params, background_process=True)
    else:
        get_latency(optimization_config=optimization_config, test_params=test_params, background_process=True)

    logger.info("[{}] process finished".format(os.getpid()))


def get_latency(optimization_config, test_params, test_result=None, background_process=False):
    onnx_session, session_name = create_inference_session(optimization_config.model_path, test_params)
    onnx_output_names = optimization_config.output_names if optimization_config.output_names else [o.name for o in onnx_session.get_outputs()]

    # warmup
    for i in range(0, optimization_config.warmup_num):
        onnx_session.run(onnx_output_names, optimization_config.inference_input_dict)

    # run test
    latencies = []
    for i in range(0, optimization_config.test_num):
        t = time.perf_counter()
        onnx_session.run(onnx_output_names, optimization_config.inference_input_dict)
        latencies.append(time.perf_counter() - t)

    if not background_process:
        test_result["test_name"] = session_name

        if test_params:
            test_result["execution_provider"] = test_params.get("execution_provider")
            test_result["env_vars"] = {
                "OMP_WAIT_POLICY": str(os.getenv("OMP_WAIT_POLICY")),
                "OMP_NUM_THREADS": str(os.getenv("OMP_NUM_THREADS")),
                "KMP_AFFINITY": str(os.getenv("KMP_AFFINITY")),
                "OMP_MAX_ACTIVE_LEVELS": str(os.getenv("OMP_MAX_ACTIVE_LEVELS")),
                "ORT_TENSORRT_FP16_ENABLE": str(os.getenv("ORT_TENSORRT_FP16_ENABLE", 0))
            }
            test_result["session_options"] = {
                "inter_op_num_threads": str(test_params.get("inter_op_num_threads")),
                "intra_op_num_threads": str(test_params.get("intra_op_num_threads")),
                "execution_mode": str(test_params.get("execution_mode")),
                "graph_optimization_level": str(test_params.get("graph_optimization_level"))
            }

        test_result["latency_ms"] = {
            "avg": round(sum(latencies) / len(latencies) * 1000, 5),
            "latency_p50": round(np.percentile(latencies, 50) * 1000, 5),
            "latency_p75": round(np.percentile(latencies, 75) * 1000, 5),
            "latency_p90": round(np.percentile(latencies, 90) * 1000, 5),
            "latency_p95": round(np.percentile(latencies, 95) * 1000, 5),
            "latency_p99": round(np.percentile(latencies, 99) * 1000, 5),
            "latency_p999": round(np.percentile(latencies, 99.9) * 1000, 5),
        }
        logger.info("ONNX model average inference time = {} for test {}".format(test_result["latency_ms"]["avg"], session_name))
        test_result["throughput"] = 1000 / test_result["latency_ms"]["avg"] if test_result["latency_ms"]["avg"] != 0 else "null"


def get_throughput(optimization_config, test_params, test_result=None, background_process=False):
    onnx_session, session_name = create_inference_session(optimization_config.model_path, test_params)
    onnx_output_names = optimization_config.output_names if optimization_config.output_names else [o.name for o in onnx_session.get_outputs()]
    ds = Dataset(onnx_session, optimization_config.inputs_spec)

    runner = ServerRunner(onnx_session, ds, optimization_config, onnx_output_names)
    # warmup
    runner.warmup(optimization_config.warmup_num)

    # run test
    runner.start_run()
    runner.finish()

    if not background_process:
        is_valid, latency_result, throughput_result = parse_mlperf_log(optimization_config.result_path)
        if is_valid:
            test_result["test_name"] = session_name

            if test_params:
                test_result["execution_provider"] = test_params.get("execution_provider")
                test_result["env_vars"] = {
                    "OMP_WAIT_POLICY": str(os.getenv("OMP_WAIT_POLICY")),
                    "OMP_NUM_THREADS": str(os.getenv("OMP_NUM_THREADS")),
                    "KMP_AFFINITY": str(os.getenv("KMP_AFFINITY")),
                    "OMP_MAX_ACTIVE_LEVELS": str(os.getenv("OMP_MAX_ACTIVE_LEVELS")),
                    "ORT_TENSORRT_FP16_ENABLE": str(os.getenv("ORT_TENSORRT_FP16_ENABLE", 0))
                }
                test_result["session_options"] = {
                    "inter_op_num_threads": str(test_params.get("inter_op_num_threads")),
                    "intra_op_num_threads": str(test_params.get("intra_op_num_threads")),
                    "execution_mode": str(test_params.get("execution_mode")),
                    "graph_optimization_level": str(test_params.get("graph_optimization_level"))
                }

            is_valid, latency_result, throughput_result = parse_mlperf_log(optimization_config.result_path)
            test_result["latency_ms"] = latency_result
            test_result["throughput"] = throughput_result
            logger.info("Optimal QPS = {} for test {}".format(test_result["throughput"], session_name))
        else:
            logger.error("Benchmark is not valid for test {}. Please increase the expected latency".format(session_name))


def parse_mlperf_log(result_path):
    is_valid = False
    result = {}
    latency_result = {}
    fname = os.path.join(result_path, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
            m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
            if m:
                result[m.group(1).strip()] = m.group(2).strip()
    throughput_result = float(result.get("Scheduled samples per second"))

    def parse_latency_ms(n):
        return float(result[n]) / 1000000

    latency_result["avg"] = parse_latency_ms("Mean latency (ns)")
    latency_result["latency_p50"] = parse_latency_ms("50.00 percentile latency (ns)")
    latency_result["latency_p90"] = parse_latency_ms("90.00 percentile latency (ns)")
    latency_result["latency_p95"] = parse_latency_ms("95.00 percentile latency (ns)")
    latency_result["latency_p99"] = parse_latency_ms("99.00 percentile latency (ns)")
    latency_result["latency_p99.9"] = parse_latency_ms("99.90 percentile latency (ns)")

    return is_valid, latency_result, throughput_result
