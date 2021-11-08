import numpy as np
import logging
import time
from server_runner import ServerRunner
import itertools
import psutil
from mlperf_dataset import Dataset
import os
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimal_qps(optimization_config):
    all_test_results = []
    for tuning_combo in generate_tuning_combos(optimization_config):
        all_test_results.extend(threads_num_tuning(optimization_config, tuning_combo))
    return all_test_results

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

    if not tuning_results:
        logger.error("Tuning process failed for ep {}".format(provider))

    return tuning_results


def threads_num_binary_search(optimization_config, test_params, tuning_results, cpu_cores):
    threads_names = ["inter_op_num_threads", "intra_op_num_threads"]

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

        test_result = get_throughput(optimization_config, test_params)
        if test_result:
            tuning_results.append(test_result)
        optimial_qps = test_result["qps"]

        current_threads_num = upper_threads_num

        while lower_threads_num < upper_threads_num:
            test_params[threads_name] = current_threads_num

            test_result = get_throughput(optimization_config, test_params)
            if test_result:
                tuning_results.append(test_result)

            mid_threads_num = lower_threads_num + (upper_threads_num - lower_threads_num) // 2
            if optimial_qps > test_result["qps"]:
                upper_threads_num = mid_threads_num
                current_threads_num = upper_threads_num
            else:
                lower_threads_num = mid_threads_num + 1
                best_threads_num = current_threads_num
                current_threads_num = lower_threads_num

        test_params[threads_name] = best_threads_num


def generate_tuning_combos(optimization_config):
    tuning_combos = itertools.product(optimization_config.omp_wait_policy_list, optimization_config.kmp_affinity,
                                      optimization_config.omp_max_active_levels, optimization_config.providers_list,
                                      optimization_config.execution_mode_list, optimization_config.ort_opt_level_list)
    yield from tuning_combos


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


def get_throughput(optimization_config, test_params):
    onnx_session, session_name = create_inference_session(optimization_config.model_path, test_params)
    onnx_output_names = optimization_config.output_names if optimization_config.output_names else [o.name for o in onnx_session.get_outputs()]
    ds = Dataset(onnx_session)

    runner = ServerRunner(onnx_session, ds, optimization_config, onnx_output_names)
    runner.warmup(optimization_config.max_latency, optimization_config.max_latency_percentile)

    start_time = time.time()
    runner.start_run()
    runner.finish()
    time_taken = time.time() - start_time

    print("time_taken: ", time_taken)

    parse_mlperf_log(optimization_config.result_path)


def parse_mlperf_log(result_path):
    pass

def make_batch(self, be, id_list):
    # collect data and make it a batch
    # for data dim=0 is batch, dim=1 is the input nr - we need to swap them
    # data = np.swapaxes(data, 0, 1)
    feed = {}
    try:
        odd_shape = be.input_shapes()
    except:
        odd_shape = self.image_size
        if len(odd_shape) != 4:
            odd_shape = [len(id_list)]+odd_shape
        assert odd_shape is not None and len(odd_shape)
        odd_shape = {be.inputs[0]: odd_shape}

    for i, name in enumerate(be.inputs):
        feed[name] = np.array([self.data_x_inmemory[id][i] for id in id_list])
        if len(odd_shape[name]) != len(feed[name].shape):
            feed[name] = np.squeeze(feed[name], axis=0)
    return feed
