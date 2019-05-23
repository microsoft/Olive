import argparse
import json
import glob
import os
import shutil
import subprocess
import sys
import uuid

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
        
        self.model = args.model
        self.result_dir = args.result

        self.args = args

        self.command = None
    
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



def run_perf_test(test_params, percentiles=False):
    print()

    result_file = os.path.join(test_params.result_dir, str(uuid.uuid4()))

    if percentiles:
        test_args = test_params.get_percentiles_args(result_file)
    else:
        test_args = test_params.get_args(result_file)
    
    perf_test = subprocess.run(test_args, capture_output=True, env=test_params.env)
    # The first run was warmup.
    remove(result_file)
    test_params.print_args(test_args)
    if not test_params.command:
        test_params.command = " ".join(test_args)
    perf_test = subprocess.run(test_args, env=test_params.env)
    latencies = []
    if perf_test.returncode == 0 and os.path.exists(result_file):
        with open(result_file) as result:
            for line in result.readlines():
                line = line.strip()
                tokens = line.split(",")
                if len(tokens) == 5:
                    latencies.append(float(tokens[1]))

        if not percentiles:
            # Profile in case of success
            profile_name = os.path.join(test_params.result_dir, str(uuid.uuid4()))
            perf_test = subprocess.run(test_params.get_profile_args(profile_name, result_file), capture_output=True, env=test_params.env)
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

def parse_arguments():
    parser = argparse.ArgumentParser()

    cores = os.cpu_count() // 2
    print("Cores: ", cores)

    parser.add_argument("--config", default="RelWithDebInfo",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to run.")
    parser.add_argument("-m", default="times",
                        choices=["duration", "times"],
                        help="Specifies the test mode. Value coulde be 'duration' or 'times'.")
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
    parser.add_argument("model",
                        help="Model.")
    parser.add_argument("result",
                        help="Result folder.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    providers = [p for p in args.e.split(",") if p != ""]
    
    bin_dir = os.path.join(os.path.dirname(__file__), "bin", args.config)
    build_dirs = os.listdir(bin_dir)
    tests = []
    for build_name in build_dirs:
        build_path = os.path.join(bin_dir, build_name)
        if os.path.isdir(build_path):
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
                    for cores in range(int(args.x), 1, -1):
                        if "openmp" in build_name:
                            params = PerfTestParams(
                                build_name + "_" + str(cores) + "_cores_passive",
                                build_name + " " + str(cores) + " cores, passive",
                                build_path,
                                test_args + ["-x", str(cores)],
                                {"OMP_WAIT_POLICY": "PASSIVE"},
                                args)
                            tests.append(params)

                            params = PerfTestParams(
                                build_name + "_" + str(cores) + "_cores_active",
                                build_name + " " + str(cores) + " cores, active",
                                build_path,
                                test_args + ["-x", str(cores)],
                                {"OMP_WAIT_POLICY": "ACTIVE"},
                                args)
                            tests.append(params)
                        else:
                            params = PerfTestParams(
                                build_name + "_" + str(cores) + "_cores",
                                build_name + " " + str(cores) + " cores",
                                build_path,
                                test_args + ["-x", str(cores)],
                                {},
                                args)
                            tests.append(params)

                if "openmp" in build_name and args.n:
                    # parallel
                    for cores in range(int(args.n), 0, -1):
                        params = PerfTestParams(
                            build_name + "_OMP_NUM_THREADS_" + str(cores) + "_passive",
                            build_name + " OMP_NUM_THREADS=" + str(cores) + ", passive",
                            build_path,
                            test_args,
                            {"OMP_WAIT_POLICY": "PASSIVE", "OMP_NUM_THREADS": str(cores)},
                            args)
                        tests.append(params)

                        params = PerfTestParams(
                            build_name + "_OMP_NUM_THREADS_" + str(cores) + "_active",
                            build_name + " OMP_NUM_THREADS=" + str(cores) + ", active",
                            build_path,
                            test_args,
                            {"OMP_WAIT_POLICY": "ACTIVE", "OMP_NUM_THREADS": str(cores)},
                            args)
                        tests.append(params)

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

    successful = sorted([x for x in tests if x.avg], key=lambda e:e.avg)
    # Re-run fastest tests to calculate percentiles.
    for test in successful[:int(args.s)]:
        run_perf_test(test, percentiles=True)
    
    failed = [x for x in tests if not x.avg]

    print("")
    print("Results:")
    out_json = []
    with open(os.path.join(args.result, "latencies.txt"), "w") as out_file:
        for test in successful:
            print(test.name, test.avg, "s")
            print(test.name, test.avg, "s", file=out_file)

            json_record = dict()
            json_record["name"] = test.name
            json_record["command"] = test.command
            json_record["avg"] = test.avg
            json_record["min"] = test.latencies[0]
            num_latencies = len(test.latencies)
            json_record["max"] = test.latencies[num_latencies - 1]
            json_record["p50"] = round((test.latencies[num_latencies // 2] + test.latencies[(num_latencies - 1) // 2]) / 2, 9)
            if num_latencies >= 4:
                json_record["p75"] = test.latencies[int(num_latencies * .75)]
            if num_latencies >= 10:
                json_record["p90"] = test.latencies[int(num_latencies * .9)]
            if num_latencies >= 20:
                json_record["p95"] = test.latencies[int(num_latencies * .95)]
            if num_latencies >= 100:
                json_record["p99"] = test.latencies[int(num_latencies * .99)]

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
