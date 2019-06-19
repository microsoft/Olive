import argparse
import json
import glob
import os
import shutil
import subprocess
import sys
import uuid
import psutil
from monitor import Monitor
from maps import ep_envvar_map

# FNULL = open(os.devnull, 'w')
def is_windows():
    return sys.platform.startswith("win")

def remove(file):
    if os.path.exists(file):
        os.remove(file)


class PerfTestParams:
    def __init__(self, name, desc, path, test_args, env, args):
        if is_windows():
            self.exe = os.path.join(build_path, "onnxruntime_perf_test.exe")
        else:
            self.exe = os.path.join(build_path, "onnxruntime_perf_test")
            env["LD_LIBRARY_PATH"] = build_path

        self.name = name
        self.desc = desc
        self.path = path
        self.test_args = [self.exe] + test_args
        
        self.env = os.environ.copy()
        self.env.pop("OMP_WAIT_POLICY", None)
        self.env.pop("OMP_NUM_THREADS", None)
        self.env.update(env)
        self.env_to_set = env
        
        self.model = args.model
        self.result_dir = args.result

        self.args = args

        self.command = None
        self.avg = None

        self.gpu = 0
        self.cpu = 0
    
    def get_common_args(self):
        common_args = []
        if self.args.m:
            common_args = common_args + ["-m", self.args.m]
        if args.m == "times":
            if self.args.r:
                common_args = common_args + ["-r", args.r]
        else:
            if self.args.t:
                common_args = common_args + ["-t", args.t]
        return common_args

    def get_args(self, result_file):
        return self.test_args + self.get_common_args() + [self.model, result_file]

    def get_percentiles_args(self, result_file):
        return self.test_args + ["-m", "times", "-r", "200", self.model, result_file]

    def get_profile_args(self, profile_file, result_file):
        return self.test_args + self.get_common_args() + ["-p", profile_file, self.model, result_file]

    def print_args(self, args):
        if self.env.get("OMP_WAIT_POLICY"):
            print("OMP_WAIT_POLICY=" + self.env["OMP_WAIT_POLICY"])
        if self.env.get("OMP_NUM_THREADS"):
            print("OMP_NUM_THREADS=" + self.env["OMP_NUM_THREADS"])
        print(" ".join(args))

    def gen_code_snippet(self):
        # import onnxruntime as rt
        # so = rt.SessionOptions()
        code_snippet = {
            "execution_provider": self.args.e,
            "environment_variables": self.env_to_set,
            "code": "\
                import onnxruntime as ort \
                so = rt.SessionOptions() \
                so.set_graph_optimization_level(2) \
                so.enable_sequential_execution = {} \
                so.session_thread_pool_size({}) \
                session = rt.Session(\"{}\", so) \
                ".format(True if self.args.x == 1 else False, self.args.x, self.args.model)
        }
        return code_snippet


def run_perf_test(test_params, percentiles=False):
    print()

    result_file = os.path.join(test_params.result_dir, str(uuid.uuid4()))

    if percentiles:
        test_args = test_params.get_percentiles_args(result_file)
    else:
        test_args = test_params.get_args(result_file)
    
    perf_test = subprocess.run(test_args, env=test_params.env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
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
            m = Monitor(latencies[0] // 20)
            perf_test = subprocess.run(test_params.get_profile_args(profile_name, result_file), 
                env=test_params.env,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            m.stop()
            print("cpu = ", m.recorded_cpu)
            m.recorded_gpu = [x for x in m.recorded_gpu if not x == 0]
            print(m.recorded_gpu)
            test_params.gpu = sum(m.recorded_gpu)/len(m.recorded_gpu) if len(m.recorded_gpu) > 0 else 0
            test_params.cpu = sum(m.recorded_cpu) / len(m.recorded_cpu) if len(m.recorded_cpu) > 0 else 0
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
        test_params.avg = round(sum(latencies) / len(latencies), 9)
    else:
        test_params.avg = None

    print(test_params.name, test_params.avg)

def run_perf_test_binary(test_params, num_cores, name_suffix, desc_suffix, env, failed_tests, successful_tests):
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
        test_params.env,
        test_params.args
    )
    if len(env) == 0:
        # tune threads by args
        param.test_args = test_params.test_args + ["-x", str(lower)]
    else:
        # tune threads by env variables
        param.env[env] = str(lower)
    run_perf_test(param)
    if not param.avg:
        # TODO: fall back to sequential???
        failed_tests.append(param)
        return
    else:
        successful_tests.append(param)
        best_latency = param.avg
        best_run = lower
    
    while lower < upper:
        mid = lower + (upper - lower) // 2
        print("lower: %d, mid: %d, upper: %d" % (lower, mid, upper))
        # Run perf test
        param = PerfTestParams(
            test_params.name + str(mid) + name_suffix,
            test_params.desc + str(mid) + desc_suffix,
            test_params.path,
            [],
            test_params.env,
            test_params.args
        )
        if len(env) == 0:
            # tune threads by args
            param.test_args = test_params.test_args + ["-x", str(mid)]
        else:
            # tune threads by env variables
            param.env[env] = str(mid)
        run_perf_test(param)
        if param.avg:
            print("current latency: ", param.avg)
            print("best latency: ", best_latency)
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
        if lower >= upper and best_run == (1 + num_cores) // 2:
            # Re-run search on the first half 
            upper = mid - 1
            lower = 1

def select_graph_optimizer(ep):
    # Placeholder function for graph optimizer
    return

def select_ep(model):
    # Placeholder for selecting appropriate ep
    return 

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
        self.m = loaded_json["mode"] if loaded_json.get("mode") else "times"
        self.e = loaded_json["execution_provider"] if loaded_json.get("execution_provider") else ""
        self.r = loaded_json["repeated_times"] if loaded_json.get("repeated_times") else "20"
        self.t = loaded_json["duration_time"] if loaded_json.get("duration_time") else "10"
        self.x = loaded_json["parallel"] if loaded_json.get("parallel") else ""
        self.n = loaded_json["num_threads"] if loaded_json.get("num_threads") else str(cores)
        self.s = loaded_json["top_n"] if loaded_json.get("top_n") else "5"

def parse_arguments():
    parser = argparse.ArgumentParser()

    cores = os.cpu_count() // 2
    print("Cores: ", cores)
    parser.add_argument("--input_json", 
                        help="A JSON file specifying the run specs. ")
    parser.add_argument("--config", default="RelWithDebInfo",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to run.")
    parser.add_argument("-m", default="times",
                        choices=["duration", "times"],
                        help="Specifies the test mode. Value could be 'duration' or 'times'.")
    parser.add_argument("-e", default="",
                        help="Specifies the provider 'cpu','cuda','mkldnn'.")
    parser.add_argument("-r", default="20",
                        help="Specifies the repeated times if running in 'times' test mode. Default:20.")
    parser.add_argument("-t", default="10",
                        help="Specifies the seconds to run for 'duration' mode. Default:10.")
    parser.add_argument("-x",
                        help="Use parallel executor, default (without -x): sequential executor.")
    parser.add_argument("-n", default=str(cores),
                        help="OMP_NUM_THREADS value.")
    parser.add_argument("-s", default="5",
                        help="Show percentiles for top n runs. Default:5.")
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

    providers = [p for p in args.e.split(",") if p != ""]
    
    bin_dir = os.path.join(os.path.dirname(__file__), "bin", args.config)
    build_dirs = os.listdir(bin_dir)
    tests = []
    successful = []
    failed = []
    # Loop for every pre-built execution provider
    for build_name in build_dirs:
        build_path = os.path.join(bin_dir, build_name)
        if os.path.isdir(build_path):
            # If current build is requested by user, run perf tuning
            if build_name in providers or len(providers) == 0:
                test_args = []

                if "mkldnn" in build_name:
                    test_args = test_args + ["-e", "mkldnn"]
                if "cuda" in build_name:
                    test_args = test_args + ["-e", "cuda"]
                if "tensorrt" in build_name:
                    test_args = test_args + ["-e", "tensorrt"]
                if "ngraph" in build_name:
                    test_args = test_args + ["-e", "ngraph"]

                if args.x:
                    # parallel
                    # for cores in range(int(args.x), 1, -1):
                    if "openmp" in build_name:
                        # Tune OMP_WAIT_POLICY passive and active 
                        # TODO: KMP_BLOCKTIME
                        run_perf_test_binary(
                            PerfTestParams(
                                build_name + "_",
                                build_name + " ",
                                build_path,
                                test_args,
                                {"OMP_WAIT_POLICY": "PASSIVE"},
                                args
                            ), int(args.x), "_cores_passive", " cores, passive", "", failed, successful)

                        run_perf_test_binary(
                            PerfTestParams(
                                build_name + "_",
                                build_name + " ",
                                build_path,
                                test_args,
                                {"OMP_WAIT_POLICY": "ACTIVE"},
                                args
                            ), int(args.x), "_cores_active", " cores, active", "", failed, successful)
                    else:
                        run_perf_test_binary(
                            PerfTestParams(
                                build_name + "_",
                                build_name + " ",
                                build_path,
                                test_args,
                                {},
                                args
                            ), int(args.x), "_cores", " cores", "", failed, successful)

                if "openmp" in build_name and args.n:
                    # parallel
                    # Wait for OMP_NUM_THREADS moving to session option. 
                    # binary tuning OMP_NUM_THREADS from 1 to args.n
                    # Passive
                    run_perf_test_binary(
                        PerfTestParams(
                            build_name + "_OMP_NUM_THREADS_",
                            build_name + " OMP_NUM_THREADS=",
                            build_path,
                            test_args,
                            {"OMP_WAIT_POLICY": "PASSIVE"},
                            args
                        ), int(args.n), "_passive", ", passive", "OMP_NUM_THREADS", failed, successful)
                    # Active
                    run_perf_test_binary(
                        PerfTestParams(
                            build_name + "_OMP_NUM_THREADS_",
                            build_name + " OMP_NUM_THREADS=",
                            build_path,
                            test_args,
                            {"OMP_WAIT_POLICY": "ACTIVE"},
                            args
                        ), int(args.n), "_active", ", active", "OMP_NUM_THREADS", failed, successful)

                # sequential
                if "openmp" in build_name:
                    params = PerfTestParams(
                        build_name + "_passive",
                        build_name + " passive",
                        build_path,
                        test_args,
                        {"OMP_WAIT_POLICY": "PASSIVE"},
                        args)
                    tests.append(params)

                    params = PerfTestParams(
                        build_name + "_active",
                        build_name + " active",
                        build_path,
                        test_args,
                        {"OMP_WAIT_POLICY": "ACTIVE"},
                        args)
                    tests.append(params)
                else:
                    params = PerfTestParams(
                        build_name,
                        build_name,
                        build_path,
                        test_args,
                        {},
                        args)
                    tests.append(params)

    # Run the tests.
    for test in tests:
        run_perf_test(test)

    successful.extend([x for x in tests if x.avg])
    successful = sorted(successful, key=lambda e:e.avg)
    # Re-run fastest tests to calculate percentiles.
    for test in successful[:int(args.s)]:
        run_perf_test(test, percentiles=True)
    
    failed.extend([x for x in tests if not x.avg])
    
    # Re-sort tests based on 100 runs
    successful = sorted(successful, key=lambda e:e.avg)
    print("")
    print("Results:")
    out_json = []
    with open(os.path.join(args.result, "latencies.txt"), "w") as out_file:
        for test in successful[:int(args.s)]:
            print(test.name, test.avg, "s")
            print(test.name, test.avg, "s", file=out_file)

            json_record = dict()
            json_record["name"] = test.name
            json_record["command"] = test.command
            json_record["avg"] = test.avg
            # json_record["min"] = test.latencies[0]
            num_latencies = len(test.latencies)
            # json_record["max"] = test.latencies[num_latencies - 1]
            # json_record["p50"] = round((test.latencies[num_latencies // 2] + test.latencies[(num_latencies - 1) // 2]) / 2, 9)
            # if num_latencies >= 4:
            #     json_record["p75"] = test.latencies[int(num_latencies * .75)]
            if num_latencies >= 10:
                json_record["p90"] = test.latencies[int(num_latencies * .9)]
            if num_latencies >= 20:
                json_record["p95"] = test.latencies[int(num_latencies * .95)]
            json_record["cpu_usage"] = test.cpu / 100
            json_record["gpu_usage"] = test.gpu
            json_record["code_snippet"] = test.gen_code_snippet()
            # if num_latencies >= 100:
            #     json_record["p99"] = test.latencies[int(num_latencies * .99)]

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
