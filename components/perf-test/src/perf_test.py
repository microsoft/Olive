import argparse
import json
import glob
import os
import shutil
import subprocess
import sys
import uuid
import psutil
import GPUtil
from monitor import Monitor
from maps import ep_envvar_map, ep_graphOptimizer_map, model_ep_map

# FNULL = open(os.devnull, 'w')
def is_windows():
    return sys.platform.startswith("win")

def remove(file):
    if os.path.exists(file):
        os.remove(file)


class PerfTestParams:
    def __init__(self, name, desc, path, test_args, env, args, build_name, thread=0):
        
        self.name = name
        self.desc = desc
        self.path = path
        
        self.env = os.environ.copy()
        if is_windows():
            self.exe = os.path.join(build_path, "onnxruntime_perf_test.exe")
        else:
            self.exe = os.path.join(build_path, "onnxruntime_perf_test")
            self.env["LD_LIBRARY_PATH"] = build_path
        
        self.test_args = [self.exe] + test_args
        self.env.pop("OMP_WAIT_POLICY", None)
        self.env.pop("OMP_NUM_THREADS", None)
        self.env.update(env)
        self.env_to_set = env
        
        self.model = args.model
        self.result_dir = args.result

        self.args = args
        self.build_name = build_name

        self.command = None
        self.avg = None

        self.gpu = 0
        self.cpu = 0
        self.memory = 0
        self.thread = 0

    def get_common_args(self):
        common_args = ["-o", self.args.optimization_level]
        if self.args.mode:
            common_args = common_args + ["-m", self.args.mode]
        if args.mode == "times":
            if self.args.repeated_times:
                common_args = common_args + ["-r", args.repeated_times]
        else:
            if self.args.duration_time:
                common_args = common_args + ["-t", args.duration_time]
        return common_args

    def get_args(self, result_file):
        return self.test_args + self.get_common_args() + [self.model, result_file]

    def get_percentiles_args(self, result_file):
        return self.test_args + ["-o", self.args.optimization_level, "-m", "times", "-r", "200", self.model, result_file]

    def get_profile_args(self, profile_file, result_file):
        return self.test_args + self.get_common_args() + ["-p", profile_file, self.model, result_file]

    def print_args(self, args):
        if self.env.get("OMP_WAIT_POLICY"):
            print("OMP_WAIT_POLICY=" + self.env["OMP_WAIT_POLICY"])
        if self.env.get("OMP_NUM_THREADS"):
            print("OMP_NUM_THREADS=" + self.env["OMP_NUM_THREADS"])
        print(" ".join(args))

    def gen_code_snippet(self):
        code_snippet = {
            "execution_provider": self.build_name,
            "environment_variables": self.env_to_set,
            "code": "\
                import onnxruntime as ort \
                so = rt.SessionOptions() \
                so.set_graph_optimization_level({}) \
                so.enable_sequential_execution = {} \
                so.session_thread_pool_size({}) \
                session = rt.Session(\"{}\", so) \
                ".format(self.args.optimization_level, False if self.args.parallel else True, self.thread, self.args.model)
        }
        return code_snippet


def run_perf_test(test_params, percentiles=False):
    print()

    result_file = os.path.join(test_params.result_dir, str(uuid.uuid4()))

    if percentiles:
        test_args = test_params.get_percentiles_args(result_file)
    else:
        test_args = test_params.get_args(result_file)
    perf_test = subprocess.run(test_args, env=test_params.env)
    # The first run was warmup.
    remove(result_file)
    test_params.print_args(test_args)
    if not test_params.command:
        test_params.command = " ".join(test_args)
    perf_test = subprocess.run(test_args, env=test_params.env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    latencies = []
    if perf_test.returncode == 0 and os.path.exists(result_file):
        with open(result_file) as result:
            for line in result.readlines():
                line = line.strip()
                tokens = line.split(",")
                if len(tokens) == 5:
                    latencies.append(float(tokens[1]))

        if percentiles:
            # Profile in case of success
            profile_name = os.path.join(test_params.result_dir, str(uuid.uuid4()))
            # print("memory info ", psutil.virtual_memory().percent)
            m = Monitor(latencies[0])
            perf_test = subprocess.run(test_params.get_profile_args(profile_name, result_file), 
                env=test_params.env,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            m.stop()
            m.recorded_gpu = [x for x in m.recorded_gpu if not x == 0]
            m.recorded_cpu = [x for x in m.recorded_cpu if not x == 0]            
            # print("cpu = ", m.recorded_cpu)
            # print(m.recorded_gpu)
            # print(m.recorded_memory)
            test_params.gpu = sum(m.recorded_gpu)/len(m.recorded_gpu) if len(m.recorded_gpu) > 0 else 0
            test_params.cpu = sum(m.recorded_cpu) / len(m.recorded_cpu) if len(m.recorded_cpu) > 0 else 0
            test_params.memory = sum(m.recorded_memory) / len(m.recorded_memory) if len(m.recorded_memory) > 0 else 0
            if perf_test.returncode == 0:
                # Find profile result
                files = glob.glob(profile_name + "*")
                if len(files) == 1:
                    with open(files[0]) as profile:
                        with open(os.path.join(test_params.result_dir, "profile_" + test_params.name + ".tsv"), "w") as out_profile:
                            print("cat\tpid\ttid\tdur\tts\tph\tname\targs", file=out_profile)
                            for record in json.load(profile):
                                print(str(record["cat"]) + "\t" + str(record["pid"]) + "\t" + str(record["tid"]) + "\t" + str(record["dur"]) + "\t" + str(record["ts"]) + "\t" + str(record["ph"]) + "\t" + str(record["name"]) + "\t" + str(record["args"]), file=out_profile)
                    shutil.move(files[0], os.path.join(test_params.result_dir, "profile_" + test_params.name + ".json"))

    remove(result_file)

    latencies.sort()

    test_params.latencies = latencies

    if len(latencies) > 0:
        test_params.avg = round(sum(latencies) / len(latencies), 9) * 1000
    else:
        test_params.avg = None

    print(test_params.name, test_params.avg)

def run_perf_test_binary(test_params, num_cores, name_suffix, desc_suffix, failed_tests, successful_tests, is_omp=False):
    lower = 1
    upper = num_cores
    mid = lower + (upper - lower) // 2
    if lower > upper:
        return
    best_latency = float("inf")
    best_run = -1

    # start latency
    param = PerfTestParams(
        test_params.name + str(lower) + name_suffix,
        test_params.desc + str(lower) + desc_suffix,
        test_params.path,
        [],
        test_params.env_to_set,
        test_params.args,
        build_name,
        lower
    )
    # tune threads by args
    if not is_omp:
        param.test_args = test_params.test_args + ["-x", str(lower)]
    else:
        param.env.update({"OMP_NUM_THREADS": str(lower)})

    run_perf_test(param)
    if not param.avg:
        # TODO: fall back to sequential???
        failed_tests.append(param)
        return
    else:
        successful_tests.append(param)
        best_latency = param.avg
        best_run = lower
    
    # Start binary search best thread pool size candidate
    lower += 1
    while lower <= upper:
        mid = lower + (upper - lower) // 2
        # print("lower: %d, mid: %d, upper: %d" % (lower, mid, upper))
        # Run perf test
        param = PerfTestParams(
            test_params.name + str(mid) + name_suffix,
            test_params.desc + str(mid) + desc_suffix,
            test_params.path,
            [],
            test_params.env_to_set,
            test_params.args,
            build_name,
            mid
        )
        # tune threads by args
        if not is_omp:
            param.test_args = test_params.test_args + ["-x", str(mid)]
        else:
            param.env.update({"OMP_NUM_THREADS": str(mid)})
        run_perf_test(param)
        if param.avg:
            # print("current latency: ", param.avg)
            # print("best latency: ", best_latency)
            successful_tests.append(param)
            if best_latency < param.avg:
                upper = mid - 1
            else:
                lower = mid + 1
                best_latency = param.avg
                best_run = mid
        else:
            failed_tests.append(param)
            break
        if lower > upper and best_run == (1 + num_cores) // 2:
            # Re-run search on the first half if the best latency lies in the middle
            upper = (1 + num_cores) // 2 - 1
            lower = 2
    return best_run

def get_env_var_combos(env_vars):
    if env_vars == None:
        return [[""]]
    import itertools
    env_options_list = env_vars.values()
    all_combos = []
    for options in env_options_list:
        if len(all_combos) == 0:
            all_combos = list(itertools.combinations(options + [""], 1))
        else:
            nested_combo = list(itertools.product(all_combos, options + [""]))
            # flatten if necessary
            all_combos = []
            for combo in nested_combo:
                a = []
                for el in combo:
                    if isinstance(el, tuple) or isinstance(el, list) :
                        a.extend(el)
                    else:
                        a.append(el)
                all_combos.append(a)
    # print(all_combos)
    return all_combos

# env_names: a list of environment variable names
# env_options: the environment variable value corresponding to the env_names
def gen_env_var_dict(env_names, env_option):
    if len(env_names) == 0 or len(env_option) == 0:
        return {}
    env_var_dict = {}
    for i in range(0, len(env_names)):
        if len(str(env_option[i])) > 0:
            env_var_dict.update({env_names[i]: str(env_option[i])})
    return env_var_dict

def select_graph_optimizer(ep):
    # Placeholder function for graph optimizer
    return ep_graphOptimizer_map.get(ep)

def select_ep(model):
    # Placeholder for selecting appropriate ep
    return model_ep_map.get(model)

class ConverterParamsFromJson():
    def __init__(self):
        with open(parse_arguments().input_json) as f:
            loaded_json = json.load(f)
        cores = os.cpu_count() // 2
        # Check the required inputs
        if loaded_json.get("model") == None:
            raise ValueError("Please specified \"model\" in the input json. ")
        if loaded_json.get("result") == None:
            raise ValueError("Please specified \"result_path\" in the input json. ")
        if loaded_json.get("config") and ["Debug", "MinSizeRel", "Release", "RelWithDebInfo"].index(loaded_json["config"]) == -1:
            raise ValueError("Please specify config with one of the following -\"Debug\", \"MinSizeRel\", \"Release\", \"RelWithDebInfo\"")
        if loaded_json.get("mode") and ["duration", "times"].index(loaded_json["mode"]) == -1:
            raise ValueError("Please specify mode with one of the following - \"duration\", \"times\"")
        self.model = loaded_json["model"]
        self.result = loaded_json["result"]
        self.config = loaded_json["config"] if loaded_json.get("config") else "RelWithDebInfo"
        self.mode = loaded_json["mode"] if loaded_json.get("mode") else "times"
        self.execution_provider = loaded_json["execution_provider"] if loaded_json.get("execution_provider") else ""
        self.repeated_times = loaded_json["repeated_times"] if loaded_json.get("repeated_times") else "20"
        self.duration_time = loaded_json["duration_time"] if loaded_json.get("duration_time") else "10"
        self.threadpool_size = loaded_json["threadpool_size"] if loaded_json.get("threadpool_size") else str(cores)
        self.num_threads = loaded_json["num_threads"] if loaded_json.get("num_threads") else str(cores)
        self.top_n = loaded_json["top_n"] if loaded_json.get("top_n") else "5"
        self.parallel = loaded_json.get["parallel"] if loaded_json.get("parallel") else True
        self.optimization_level = loaded_json["optimization_level"] if loaded_json.get("optimization_level") else "3"

def parse_arguments():
    parser = argparse.ArgumentParser()

    cores = os.cpu_count() // 2
    print("Cores: ", cores)
    parser.add_argument("--input_json", 
                        help="A JSON file specifying the run specs. ")
    parser.add_argument("--config", default="RelWithDebInfo",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to run.")
    parser.add_argument("-m", "--mode", default="times",
                        choices=["duration", "times"],
                        help="Specifies the test mode. Value could be 'duration' or 'times'.")
    parser.add_argument("-e", "--execution_provider", default="",
                        help="Specifies the provider 'cpu','cuda','mkldnn'.")
    parser.add_argument("-r", "--repeated_times", default="20",
                        help="Specifies the repeated times if running in 'times' test mode. Default:20.")
    parser.add_argument("-t", "--duration_times", default="10",
                        help="Specifies the seconds to run for 'duration' mode. Default:10.")
    parser.add_argument("-x", "--threadpool_size", default=str(cores),
                        help="Use parallel executor, default (without -x): sequential executor.")
    parser.add_argument("-n", "--num_threads", default=str(cores),
                        help="OMP_NUM_THREADS value.")
    parser.add_argument("-s", "--top_n", default="5",
                        help="Show percentiles for top n runs. Default:5.")
    parser.add_argument("-P", "--parallel", default=True,
                        help="Use parallel executor instead of sequential executor.")
    parser.add_argument("-o", "--optimization_level", default="3", 
                        help="0: disable optimization, 1: basic optimization, 2: extended optimization, 3: extended+layout optimization.")
    parser.add_argument("--model",
                        help="Model.")
    parser.add_argument("--result",
                        help="Result folder.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.input_json != None and len(args.input_json) > 0:
        args = ConverterParamsFromJson()
    else:
        if not args.model or len(args.model) == 0:
            raise ValueError("Please specify the required argument \"model\" either in a json file or by --model")
        if not args.result or len(args.result) == 0:
            raise ValueError("Please specify the required argument \"result\" either in a json file or by --result")

    if not os.path.exists(args.result):
        os.mkdir(args.result)
    
    bin_dir = os.path.join(os.path.dirname(__file__), "bin", args.config)
    build_dirs = os.listdir(bin_dir)

    providers = [p for p in args.execution_provider.split(",") if p != ""] if len(args.execution_provider) > 0 else build_dirs

    if len(GPUtil.getGPUs()) == 0:
        print("No GPU found on current device. Cuda and TensorRT performance tuning might not be available. ")
        # if "cuda" in providers:
        #     providers.remove("cuda") 
        # if "tensorrt" in providers:
        #     providers.remove("tensorrt") 
    print("providers ", providers)

    tests = []
    successful = []
    failed = []
    # Loop for every pre-built execution provider
    for build_name in build_dirs:
        build_path = os.path.join(bin_dir, build_name)
        if os.path.isdir(build_path):
            # If current build is requested by user, run perf tuning
            if build_name in providers:        
                test_args = []

                if "mkldnn" in build_name:
                    test_args = test_args + ["-e", "mkldnn"]
                if "cuda" in build_name:
                    test_args = test_args + ["-e", "cuda"]
                if "tensorrt" in build_name:
                    test_args = test_args + ["-e", "tensorrt"]
                if "ngraph" in build_name:
                    test_args = test_args + ["-e", "ngraph"]
                
                env_vars = ep_envvar_map.get(build_name)
                env_var_combos = get_env_var_combos(env_vars)
                env_names = list(env_vars.keys()) if env_vars is not None else []
                # Tune all possible combinations of environment variables, including defaults
                for combo in env_var_combos:
                    # generate env var dict {env_var_name: env_var_option}. 
                    env = gen_env_var_dict(env_names, combo)
                    env_option = ""
                    for e in env:
                        env_option += "_" + e + "_" + env.get(e)

                    best_run = -1
                    is_omp = "openmp" in build_name
                    num_threads = int(args.num_threads) if is_omp else int(args.threadpool_size)
                    if args.parallel and "cuda" not in build_name and "tensorrt" not in build_name:                 
                        # Tune environment variables and thread pool size using parallel executor 
                        best_run = run_perf_test_binary(
                            PerfTestParams(
                                build_name + "_parallel_",
                                build_name + " ",
                                build_path,
                                test_args + ["-P"],
                                env,
                                args,
                                build_name,
                            ), num_threads, "_threads" + env_option, " threads, " + env_option, failed, successful, is_omp)
                    if best_run != None and best_run > 1:
                        # Run the best thread pool candidate with environment variable on sequential executor
                        param = PerfTestParams(
                            build_name + "_" + str(best_run) + "_threads" + env_option,
                            build_name + " " + str(best_run) + " threads, " + env_option,
                            build_path,
                            test_args,
                            env,
                            args, 
                            build_name
                        )
                        if is_omp:
                            param.env.update({"OMP_NUM_THREADS": str(best_run)})
                        else:
                            param.test_args += ["-x", str(best_run)]
                        tests.append(param)
                    else:
                        # Tune environment variables and thread pool size using sequential executor
                        run_perf_test_binary(
                            PerfTestParams(
                                build_name + "_",
                                build_name + " ",
                                build_path,
                                test_args,
                                env,
                                args,
                                build_name
                            ), num_threads, "_threads" + env_option, " threads, " + env_option, failed, successful, is_omp)
                    # Tune environment variables using sequential executor
                    params = PerfTestParams(
                        build_name + env_option,
                        build_name + " " + env_option,
                        build_path,
                        test_args,
                        env,
                        args,
                        build_name)
                    tests.append(params)                    

    # Run the tests.
    for test in tests:
        run_perf_test(test)

    successful.extend([x for x in tests if x.avg])
    successful = sorted(successful, key=lambda e:e.avg)
    # Re-run fastest tests to calculate percentiles.
    for test in successful[:int(args.top_n)]:
        run_perf_test(test, percentiles=True)
    
    failed.extend([x for x in tests if not x.avg])
    
    if len(GPUtil.getGPUs()) == 0:
        failed = [x for x in failed if "cuda" not in x.name and "tensorrt" not in x.name]

    # Re-sort tests based on 100 runs
    successful_not_profiled = successful[int(args.top_n):]
    successful_profiled = sorted(successful[:int(args.top_n)], key=lambda e:e.avg)
    print("")
    print("Results:")
    out_json = []


    with open(os.path.join(args.result, "latencies.txt"), "w") as out_file:
        for test in successful_profiled:
            print(test.name, test.avg, "ms")
            print(test.name, test.avg, "ms", file=out_file)

            json_record = dict()
            json_record["name"] = test.name
            json_record["command"] = test.command
            json_record["avg"] = test.avg
            num_latencies = len(test.latencies)
            if num_latencies >= 10:
                json_record["p90"] = test.latencies[int(num_latencies * .9)] * 1000
            if num_latencies >= 20:
                json_record["p95"] = test.latencies[int(num_latencies * .95)] * 1000
            json_record["cpu_usage"] = test.cpu / 100
            json_record["gpu_usage"] = test.gpu
            json_record["memory_util"] = test.memory / 100
            json_record["code_snippet"] = test.gen_code_snippet()
            out_json.append(json_record)

        for test in successful_not_profiled:
            json_record = dict()
            json_record["name"] = test.name
            json_record["command"] = test.command
            json_record["avg"] = test.avg
            num_latencies = len(test.latencies)
            if num_latencies >= 10:
                json_record["p90"] = test.latencies[int(num_latencies * .9)] * 1000
            if num_latencies >= 20:
                json_record["p95"] = test.latencies[int(num_latencies * .95)] * 1000
            out_json.append(json_record)
            
        for test in failed:
            print(test.name, "error")
            print(test.name, "error", file=out_file)

            json_record = dict()
            json_record["name"] = test.name
            json_record["command"] = test.command
            json_record["result"] = "error"
            out_json.append(json_record)
    
    with open(os.path.join(args.result, "latencies.json"), "w") as json_file:
        json.dump(out_json, json_file, indent=2)
