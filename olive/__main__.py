import argparse
import subprocess
import sys
import time

import json
import os

from .constants import OLIVE_RESULT_PATH, TEST_NUM, WARMUP_NUM, ONNX_MODEL_PATH, ONNXRUNTIME_VERSION, PYTORCH_VERSION, \
    TENSORFLOW_VERSION, PYTHON_PATH
from .web.server import start_server, stop_server, server_dependencies_installed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_opt_config(args):
    from .optimization_config import OptimizationConfig

    if args.optimization_config:
        with open(args.optimization_config, 'r') as f:
            config_dict = json.load(f)

        input_names = config_dict.get("input_names")
        input_shapes = config_dict.get("input_shapes")
        inputs_spec = None

        if config_dict.get("inputs_spec"):
            inputs_spec = config_dict.get("inputs_spec")
        else:
            if input_names and input_shapes:
                inputs_spec = dict(zip(input_names, input_shapes))

        opt_config = OptimizationConfig(
            model_path=config_dict.get("model_path"),
            inputs_spec=inputs_spec,
            output_names=config_dict.get("output_names"),
            providers_list=config_dict.get("providers_list"),
            quantization_enabled=config_dict.get("quantization_enabled", False),
            transformer_enabled=config_dict.get("transformer_enabled", False),
            transformer_args=config_dict.get("transformer_args"),
            sample_input_data_path=config_dict.get("sample_input_data_path"),
            concurrency_num=config_dict.get("concurrency_num", 1),
            kmp_affinity=config_dict.get("kmp_affinity", ["respect,none"]),
            omp_max_active_levels=config_dict.get("omp_max_active_levels", ["1"]),
            result_path=config_dict.get("result_path", OLIVE_RESULT_PATH),
            warmup_num=config_dict.get("warmup_num", WARMUP_NUM),
            test_num=config_dict.get("test_num", TEST_NUM),
            trt_fp16_enabled=config_dict.get("trt_fp16_enabled", False),
            openmp_enabled=config_dict.get("openmp_enabled", False),
            inter_thread_num_list=config_dict.get("inter_thread_num_list", [None]),
            intra_thread_num_list=config_dict.get("intra_thread_num_list", [None]),
            ort_opt_level_list=config_dict.get("ort_opt_level_list", ["all"]),
            execution_mode_list=config_dict.get("execution_mode_list"),
            omp_wait_policy_list=config_dict.get("omp_wait_policy_list"),
            throughput_tuning_enabled=config_dict.get("throughput_tuning_enabled"),
            max_latency_percentile=config_dict.get("max_latency_percentile"),
            max_latency_ms=config_dict.get("max_latency_ms"),
            dynamic_batching_size=config_dict.get("dynamic_batching_size"),
            threads_num=config_dict.get("threads_num"),
            min_duration_sec=config_dict.get("min_duration_sec")
        )

    else:
        inputs_spec = None
        if args.inputs_spec:
            inputs_spec_str = args.inputs_spec.replace("'", "\"")
            inputs_spec = json.loads(inputs_spec_str)
        else:
            if args.input_names and args.input_shapes:
                input_names = [n for n in args.input_names.split(",") if n != ""]
                input_shapes = json.loads(args.input_shapes)
                inputs_spec = dict(zip(input_names, input_shapes))

        providers_list = [p for p in args.providers_list.split(",") if p != ""] if args.providers_list else []
        output_names = [n for n in args.output_names.split(",") if n != ""] if args.output_names else []
        kmp_affinity = [k for k in args.kmp_affinity.split(",") if k != ""] if args.kmp_affinity else ["respect,none"]
        omp_max_active_levels = [o for o in args.omp_max_active_levels.split(",") if o != ""] \
            if args.omp_max_active_levels else ["1"]
        inter_thread_num_list = [int(n) for n in args.inter_thread_num_list.split(",") if n != ""] if args.inter_thread_num_list else [None]
        intra_thread_num_list = [int(n) for n in args.intra_thread_num_list.split(",") if n != ""] if args.intra_thread_num_list else [None]
        ort_opt_level_list = [level for level in args.ort_opt_level_list.split(",") if level != ""] if args.ort_opt_level_list else ["all"]
        execution_mode_list = [e for e in args.execution_mode_list.split(",") if e != ""] if args.execution_mode_list else []
        omp_wait_policy_list = [p for p in args.omp_wait_policy_list.split(",") if p != ""] if args.omp_wait_policy_list else []

        opt_config = OptimizationConfig(
            model_path=args.model_path,
            inputs_spec=inputs_spec,
            providers_list=providers_list,
            output_names=output_names,
            quantization_enabled=args.quantization_enabled,
            transformer_enabled=args.transformer_enabled,
            transformer_args=args.transformer_args,
            sample_input_data_path=args.sample_input_data_path,
            concurrency_num=args.concurrency_num if args.concurrency_num else 1,
            kmp_affinity=kmp_affinity,
            omp_max_active_levels=omp_max_active_levels,
            result_path=args.result_path if args.result_path else OLIVE_RESULT_PATH,
            warmup_num=args.warmup_num if args.warmup_num else WARMUP_NUM,
            test_num=args.test_num if args.test_num else TEST_NUM,
            trt_fp16_enabled=args.trt_fp16_enabled,
            openmp_enabled=args.openmp_enabled,
            inter_thread_num_list=inter_thread_num_list,
            intra_thread_num_list=intra_thread_num_list,
            ort_opt_level_list=ort_opt_level_list,
            execution_mode_list=execution_mode_list,
            omp_wait_policy_list=omp_wait_policy_list,
            throughput_tuning_enabled=args.throughput_tuning_enabled,
            max_latency_percentile=args.max_latency_percentile,
            max_latency_ms=args.max_latency_ms,
            dynamic_batching_size=args.dynamic_batching_size if args.dynamic_batching_size else 1,
            threads_num=args.threads_num if args.threads_num else 1,
            min_duration_sec=args.min_duration_sec if args.min_duration_sec else 10
        )

    return opt_config


def get_cvt_config(args):
    from .conversion_config import ConversionConfig
    model_framework = args.model_framework

    if args.conversion_config:
        with open(args.conversion_config, 'r') as f:
            config_dict = json.load(f)

        input_names = config_dict.get("input_names")
        input_shapes = config_dict.get("input_shapes")
        input_types = config_dict.get("input_types")
        output_names = config_dict.get("output_names")
        output_shapes = config_dict.get("output_shapes")
        output_types = config_dict.get("output_types")

        inputs_schema = None
        outputs_schema = None

        if config_dict.get("inputs_schema"):
            inputs_schema = config_dict.get("inputs_schema")
        else:
            if input_names:
                inputs_schema = []
                try:
                    for i in range (len(input_names)):
                        d = {"name": input_names[i]}
                        if input_shapes:
                            d["shape"] = input_shapes[i]
                        if input_types:
                            d["dataType"] = input_types[i]
                        inputs_schema.append(d)
                except IndexError:
                    logger.error("Given input shapes or types have different length compared with input names.")
                    raise IndexError

        if config_dict.get("outputs_schema"):
            outputs_schema = config_dict.get("outputs_schema")
        else:
            if output_names:
                outputs_schema = []
                try:
                    for i in range (len(output_names)):
                        d = {"name": output_names[i]}
                        if output_shapes:
                            d["shape"] = output_shapes[i]
                        if output_types:
                            d["dataType"] = output_types[i]
                        outputs_schema.append(d)
                except IndexError:
                    logger.error("Given output shapes or types have different length compared with output names.")
                    raise IndexError

        cvt_config = ConversionConfig(
            model_path=config_dict.get("model_path"),
            model_root_path=config_dict.get("model_root_path"),
            inputs_schema=inputs_schema,
            outputs_schema=outputs_schema,
            model_framework=model_framework,
            onnx_opset=config_dict.get("onnx_opset"),
            onnx_model_path=config_dict.get("onnx_model_path", ONNX_MODEL_PATH),
            sample_input_data_path=config_dict.get("sample_input_data_path")
        )
    else:
        inputs_schema = None
        outputs_schema = None

        if args.inputs_schema:
            inputs_schema_str = args.inputs_schema.replace("'", "\"")
            inputs_schema = json.loads(inputs_schema_str)
        else:
            input_names = [n for n in args.input_names.split(",") if n != ""] if args.input_names else None
            input_shapes = json.loads(args.input_shapes) if args.input_shapes else None
            input_types = [t for t in args.input_types.split(",") if t != ""] if args.input_types else None
            if input_names:
                inputs_schema = []
                try:
                    for i in range(len(input_names)):
                        d = {"name": input_names[i]}
                        if input_shapes:
                            d["shape"] = input_shapes[i]
                        if input_types:
                            d["dataType"] = input_types[i]
                        inputs_schema.append(d)
                except IndexError:
                    logger.error("Given input shapes or types have different length compared with input names.")
                    raise IndexError

        if args.outputs_schema:
            outputs_schema_str = args.outputs_schema.replace("'", "\"")
            outputs_schema = json.loads(outputs_schema_str)
        else:
            output_names = [n for n in args.output_names.split(",") if n != ""] if args.output_names else None
            output_shapes = json.loads(args.output_shapes) if args.output_shapes else None
            output_types = [t for t in args.output_types.split(",") if t != ""] if args.output_types else None
            if output_names:
                outputs_schema = []
                try:
                    for i in range (len(output_names)):
                        d = {"name": output_names[i]}
                        if output_shapes:
                            d["shape"] = output_shapes[i]
                        if output_types:
                            d["dataType"] = output_types[i]
                        outputs_schema.append(d)
                except IndexError:
                    logger.error("Given output shapes or types have different length compared with output names.")
                    raise IndexError

        cvt_config = ConversionConfig(
            model_path=args.model_path,
            model_root_path=args.model_root_path,
            inputs_schema=inputs_schema,
            outputs_schema=outputs_schema,
            model_framework=model_framework,
            onnx_opset=args.onnx_opset,
            onnx_model_path=args.onnx_model_path if args.onnx_model_path else ONNX_MODEL_PATH,
            sample_input_data_path=args.sample_input_data_path
        )
    return cvt_config


def optimize_in_conda_env(args):
    conda_env_name = "OLive_optimization_{}".format(str(time.time()).split(".")[0])
    logger.info("new created conda env name is {}".format(conda_env_name))

    python_version = "3.8"
    use_gpu = args.use_gpu if args.use_gpu else False
    onnxruntime_version = args.onnxruntime_version if args.onnxruntime_version else ONNXRUNTIME_VERSION
    opt_args_str = ""
    for key in args.__dict__.keys():
        if args.__dict__[key]:
            if key not in ["use_conda", "use_docker", "use_gpu", "onnxruntime_version", "func"]:
                if key in ["quantization_enabled", "transformer_enabled", "trt_fp16_enabled", "openmp_enabled", "throughput_tuning_enabled"]:
                    opt_args_str = opt_args_str + "--{} ".format(key)
                else:
                    opt_args_str = opt_args_str + "--{} {} ".format(key, args.__dict__[key])

    if sys.platform.startswith('win'):
        conda_optimization_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization", "conda_optimization.bat")
    else:
        conda_optimization_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization", "conda_optimization.sh")
    subprocess.call([conda_optimization_file, conda_env_name, python_version, str(use_gpu), opt_args_str, onnxruntime_version])


def convert_in_conda_env(args):
    conda_env_name = "OLive_conversion_{}".format(str(time.time()).split(".")[0])
    logger.info("new created conda env name is {}".format(conda_env_name))

    python_version = "3.8"
    cvt_args_str = ""
    for key in args.__dict__.keys():
        if args.__dict__[key]:
            if key not in ["use_conda", "use_docker", "func"]:
                cvt_args_str = cvt_args_str + "--{} {} ".format(key, args.__dict__[key])

    if sys.platform.startswith('win'):
        conda_conversion_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversion", "conda_conversion.bat")
    else:
        conda_conversion_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversion", "conda_conversion.sh")

    model_framework = args.model_framework
    framework_version = args.framework_version

    if not framework_version:
        if model_framework.lower() == "pytorch":
            framework_version = PYTORCH_VERSION
        else:
            framework_version = TENSORFLOW_VERSION

    subprocess.call([conda_conversion_file, conda_env_name, python_version, model_framework, framework_version, cvt_args_str])


def run_server(args):
    if server_dependencies_installed():
        try:
            start_server()
        except KeyboardInterrupt:
            stop_server()
            logger.info("OLive server has been stopped")
            sys.exit(0)
    else:
        raise ModuleNotFoundError("Packages required for OLive server are not installed. "
                                  "Please run 'olive setup --server' to install required packages first.")


def model_opt(args):
    if args.use_conda:
        optimize_in_conda_env(args)
    elif args.use_docker:
        if sys.platform.startswith('win'):
            subprocess.run("{} -m pip install pypiwin32".format(PYTHON_PATH), shell=True, stdout=subprocess.PIPE)
        from .optimization.docker_optimization import optimize_in_docker
        optimize_in_docker(args)
    else:
        logger.warning("OLive will call \"olive setup\" to setup environment first")
        from .env_setup import install_packages
        install_packages(args.onnxruntime_version, args.use_gpu)

        from .optimize import optimize
        opt_config = get_opt_config(args)
        optimize(opt_config)


def model_cvt(args):
    if args.use_conda:
        convert_in_conda_env(args)
    elif args.use_docker:
        if sys.platform.startswith('win'):
            subprocess.run("{} -m pip install pypiwin32".format(PYTHON_PATH), shell=True, stdout=subprocess.PIPE)
        from .conversion.docker_conversion import convert_in_docker
        convert_in_docker(args)

    else:
        cvt_config = get_cvt_config(args)
        logger.warning("OLive will call \"olive setup\" to setup environment first")
        from .env_setup import install_packages
        model_framework = args.model_framework
        install_packages(model_framework=model_framework, framework_version=args.framework_version)
        from .convert import convert
        convert(cvt_config)


def setup(args):
    if args.server:
        from .env_setup import install_server_dependencies
        install_server_dependencies()
        logger.info("Packages required for OLive server have been installed.")
    else:
        from .env_setup import install_packages
        install_packages(args.onnxruntime_version, args.use_gpu, args.model_framework, args.framework_version)


def main():
    parser = argparse.ArgumentParser(prog="olive", description="OLive: ONNX Go Live")
    subparsers = parser.add_subparsers(help="sub-command")

    parser_opt = subparsers.add_parser("optimize", help="model inference optimization for ONNX model")
    # arguments for model inference optimization configuration
    parser_opt.add_argument("--optimization_config", help="config.json file for optimization")
    parser_opt.add_argument("--model_path", help="model path for optimization")
    parser_opt.add_argument("--result_path", help="result directory for OLive optimization", default=OLIVE_RESULT_PATH)
    parser_opt.add_argument("--inputs_spec", help="dict of input's names and shapes")
    parser_opt.add_argument("--input_names", help="input names for onnxruntime session inference")
    parser_opt.add_argument("--input_shapes", help="input shapes for onnxruntime session inference")
    parser_opt.add_argument("--output_names", help="output names for onnxruntime session inference")
    parser_opt.add_argument("--providers_list", help="providers used for perftuning")
    parser_opt.add_argument("--trt_fp16_enabled", help="whether enable fp16 mode for TensorRT", action="store_true")
    parser_opt.add_argument("--openmp_enabled", help="whether the onnxruntime package is built with OpenMP", action="store_true")
    parser_opt.add_argument("--quantization_enabled", help="whether enable the quantization or not", action="store_true")
    parser_opt.add_argument("--transformer_enabled", help="whether enable transformer optimization", action="store_true")
    parser_opt.add_argument("--transformer_args", help="onnxruntime transformer optimizer args")
    parser_opt.add_argument("--sample_input_data_path", help="path to sample_input_data.npz")
    parser_opt.add_argument("--concurrency_num", type=int, help="tuning process concurrency number")
    parser_opt.add_argument("--kmp_affinity", help="bind OpenMP* threads to physical processing units")
    parser_opt.add_argument("--omp_max_active_levels", help="maximum number of nested active parallel regions")
    parser_opt.add_argument("--inter_thread_num_list", help="list of inter thread number for perftuning")
    parser_opt.add_argument("--intra_thread_num_list", help="list of intra thread number for perftuning")
    parser_opt.add_argument("--execution_mode_list", help="list of execution mode for perftuning")
    parser_opt.add_argument("--ort_opt_level_list", help="onnxruntime optimization level")
    parser_opt.add_argument("--omp_wait_policy_list", help="list of OpenMP wait policy for perftuning")
    parser_opt.add_argument("--warmup_num", type=int, help="warmup times for latency measurement")
    parser_opt.add_argument("--test_num", type=int, help="repeat test times for latency measurement")
    parser_opt.add_argument("--throughput_tuning_enabled", help="whether tune model for optimal throughput", action="store_true")
    parser_opt.add_argument("--max_latency_percentile", type=float, help="throughput max latency pct tile, e.g. 0.90, 0.95")
    parser_opt.add_argument("--max_latency_ms", type=float, help="max latency in pct tile in millisecond")
    parser_opt.add_argument("--dynamic_batching_size", type=int, help="max batchsize for dynamic batching")
    parser_opt.add_argument("--threads_num", type=int, help="threads num for throughput optimization")
    parser_opt.add_argument("--min_duration_sec", type=int, help="minimum duration for each run in second")

    # arguments for environment setup
    parser_opt.add_argument("--use_conda", help="run optimization in new conda env or not", action="store_true")
    parser_opt.add_argument("--use_docker", help="run optimization in docker container or not", action="store_true")
    parser_opt.add_argument("--onnxruntime_version", help="onnxruntime version used for model optimization")
    parser_opt.add_argument("--use_gpu", help="run optimization with gpu or not", action="store_true")
    parser_opt.set_defaults(func=model_opt)

    parser_cvt = subparsers.add_parser("convert", help="model conversion from original framework to ONNX")
    # arguments for model framework conversion configuration
    parser_cvt.add_argument("--conversion_config", help="config.json file for conversion")
    parser_cvt.add_argument("--model_path", help="model path for conversion")
    parser_cvt.add_argument("--model_framework", help="model original framework")
    parser_cvt.add_argument("--model_root_path", help="model folder for conversion, only for PyTorch model")
    parser_cvt.add_argument("--inputs_schema", help="input’s names, types, and shapes")
    parser_cvt.add_argument("--outputs_schema", help="output’s names, types, and shapes")
    parser_cvt.add_argument("--input_names", help="input names for model framework conversion")
    parser_cvt.add_argument("--input_shapes", help="input shapes for model framework conversion")
    parser_cvt.add_argument("--input_types", help="input types for model framework conversion")
    parser_cvt.add_argument("--output_names", help="output names for model framework conversion")
    parser_cvt.add_argument("--output_shapes", help="output shapes for model framework conversion")
    parser_cvt.add_argument("--output_types", help="output types for model framework conversion")
    parser_cvt.add_argument("--onnx_opset", help="target opset version for conversion", type=int)
    parser_cvt.add_argument("--onnx_model_path", help="ONNX model path as conversion output", default=ONNX_MODEL_PATH)
    parser_cvt.add_argument("--sample_input_data_path", help="path to sample_input_data.npz")
    # arguments for environment setup
    parser_cvt.add_argument("--use_conda", help="run conversion in new conda env or not", action="store_true")
    parser_cvt.add_argument("--use_docker", help="run conversion in docker container or not", action="store_true")
    parser_cvt.add_argument("--framework_version", help="original framework version")
    parser_cvt.set_defaults(func=model_cvt)

    parser_setup = subparsers.add_parser("setup", help="setup environment for optimization and conversion")
    parser_setup.add_argument("--onnxruntime_version", help="onnxruntime version to be used", default=ONNXRUNTIME_VERSION)
    parser_setup.add_argument("--use_gpu", help="setup environment with gpu or not", action="store_true")
    parser_setup.add_argument("--model_framework", help="model_framework to be used")
    parser_setup.add_argument("--framework_version", help="framework_version to be used")
    parser_setup.add_argument("--server", help="install required dependecies for OLive server", action="store_true")
    parser_setup.set_defaults(func=setup)

    parser_server = subparsers.add_parser("server", help="setup local server for OLive service")
    parser_server.set_defaults(func=run_server)

    options = parser.parse_args()
    if not getattr(options, 'func', None):
        parser.print_help()
        return

    options.func(options)


if __name__ == "__main__":
    main()
