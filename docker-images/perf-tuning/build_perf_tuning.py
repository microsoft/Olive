# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import glob
import os
import shutil
import subprocess
import sys

def is_windows():
    return sys.platform.startswith("win")

def copy(src_path, dest_path):
    files = glob.glob(src_path)
    for file in files:
        shutil.copy(file, dest_path, follow_symlinks=False)

def build_onnxruntime(onnxruntime_dir, config, build_args, build_name, args):
    if args.variants and not (build_name in args.variants.split(",")):
        return

    if is_windows():
        subprocess.run([os.path.join(onnxruntime_dir, "build.bat"), "--config", config, "--build_shared_lib"] + build_args, cwd=onnxruntime_dir, check=True)
        target_dir = os.path.join("bin", config, build_name)
        
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        
        copy(os.path.join(onnxruntime_dir, "build/Windows", config, config, "onnxruntime_perf_test.exe"), target_dir)
        copy(os.path.join(onnxruntime_dir, "build/Windows", config, config, "onnxruntime.dll"), target_dir)
        if "mklml" not in build_name:
            if "--use_dnnl" in build_args:
                copy(os.path.join(onnxruntime_dir, "build/Windows", config, config, "dnnl.dll"), target_dir)
            if args.use_cuda or args.use_tensorrt:
                copy(os.path.join(args.cudnn_home, "bin/cudnn*.dll"), target_dir)
            if args.use_tensorrt:
                copy(os.path.join(args.tensorrt_home, "lib/nvinfer.dll"), target_dir)
        else:
            if "--use_tvm" in build_args:
                copy(os.path.join(onnxruntime_dir, "build", "Windows", config, config, "tvm.dll"), target_dir)
            if "--use_nuphar" in build_args:
                copy(os.path.join(onnxruntime_dir, "onnxruntime", "core", "providers", "nuphar", "scripts", "symbolic_shape_infer.py"), target_dir)
    else:
        build_env = os.environ.copy()
        lib_path = os.path.join(onnxruntime_dir, "build/Linux", config, "mklml/src/project_mklml/lib/")
        build_env["LD_LIBRARY_PATH"] = lib_path
        subprocess.run([os.path.join(onnxruntime_dir, "build.sh"), "--config", config, "--build_shared_lib"] + build_args, cwd=onnxruntime_dir, check=True, env=build_env)
        target_dir = os.path.join("bin", config, build_name)
        
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        
        copy(os.path.join(onnxruntime_dir, "build/Linux", config, "onnxruntime_perf_test"), target_dir)
        copy(os.path.join(onnxruntime_dir, "build/Linux", config, "libonnxruntime.so*"), target_dir)
        copy(os.path.join(onnxruntime_dir, "build/Linux", config, "mklml/src/project_mklml/lib/*.so*"), target_dir)
        if "all_eps" in build_name:
            if "--use_dnnl" in build_args:
                copy(os.path.join(onnxruntime_dir, "build/Linux", config, "dnnl/install/lib/libdnnl.so*"), target_dir)
            if args.use_cuda or args.use_tensorrt:
                if is_windows():
                    copy(os.path.join(args.cudnn_home, "bin/cudnn*.dll"), target_dir)
                else:
                    copy(os.path.join(args.cudnn_home, "lib64/libcudnn.so*"), target_dir)
                    copy(os.path.join(args.cudnn_home, "lib64/libnvrtc.so*"), target_dir)
            if args.use_tensorrt:
                if is_windows():
                    copy(os.path.join(args.tensorrt_home, "lib/nvinfer.dll"), target_dir)
                else:
                    copy(os.path.join(args.tensorrt_home, "lib/libnvinfer.so*"), target_dir)
                    copy(os.path.join(args.tensorrt_home, "lib/libnvinfer_plugin.so*"), target_dir)
            
        if "mklml" in build_name:
            if "--use_tvm" in build_args:
                copy(os.path.join(onnxruntime_dir, "build/Linux", config, "external", "tvm", "libtvm.so*"), target_dir)
            if "--use_nuphar" in build_args:
                copy(os.path.join(onnxruntime_dir, "onnxruntime", "core", "providers", "nuphar", "scripts", "symbolic_shape_infer.py"), target_dir)
        if "--use_ngraph" in build_args:
            copy(os.path.join(onnxruntime_dir, "build/Linux", config, "external/ngraph/lib/lib*.so*"), target_dir)
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnxruntime_home", help="Path to onnxruntime home.")

    parser.add_argument("--config", default="RelWithDebInfo",
                        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
                        help="Configuration to build.")
    parser.add_argument("--use_cuda", action='store_true', help="Enable CUDA.")
    parser.add_argument("--cuda_version", help="The version of CUDA toolkit to use. Auto-detect if not specified. e.g. 9.0")
    parser.add_argument("--cuda_home", help="Path to CUDA home."
                                            "Read from CUDA_HOME environment variable if --use_cuda is true and --cuda_home is not specified.")
    parser.add_argument("--cudnn_home", help="Path to CUDNN home. "
                                             "Read from CUDNN_HOME environment variable if --use_cuda is true and --cudnn_home is not specified.")
    parser.add_argument("--use_tensorrt", action='store_true', help="Build with TensorRT")
    parser.add_argument("--tensorrt_home", help="Path to TensorRT installation dir")

    parser.add_argument("--use_ngraph", action='store_true', help="Build with nGraph")
    parser.add_argument("--use_mklml", action='store_true', help="Build with mklml")
    parser.add_argument("--use_nuphar", action='store_true', help="Build with Nuphar")
    parser.add_argument("--llvm_path", help="Path to llvm-build/lib/cmake/llvm")

    parser.add_argument("--variants", help="Variants to build. Will build all by default")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    build_args = ["--parallel", "--use_dnnl", "--use_openmp"]
    nuphar_args = []

    if args.use_nuphar:
        nuphar_args += ["--use_tvm", "--use_llvm", "--use_nuphar"]
        if args.llvm_path:
            nuphar_args = nuphar_args + ["--llvm_path", args.llvm_path]
    if args.use_ngraph:
        # Build ngraph as a separate build
        build_onnxruntime(args.onnxruntime_home, args.config, ["--parallel", "--use_ngraph", "--use_openmp"] + nuphar_args, "ngraph", args)
    if args.use_mklml:
        # Build mklml + nuphar in one build
        build_onnxruntime(args.onnxruntime_home, args.config, ["--parallel", "--use_mklml"] + nuphar_args, "mklml", args)

    if args.use_cuda:
        build_args += ["--use_cuda"]
        if args.cuda_version:
            build_args = build_args + ["--cuda_version", args.cuda_version]
        if args.cuda_home:
            build_args = build_args + ["--cuda_home", args.cuda_home]
        if args.cudnn_home:
            build_args = build_args + ["--cudnn_home", args.cudnn_home]

    if args.use_tensorrt:
        build_args += ["--use_tensorrt", "--use_full_protobuf"]
        if args.tensorrt_home:
            build_args = build_args + ["--tensorrt_home", args.tensorrt_home]
        if not args.use_cuda:
            if args.cuda_version:
                build_args = build_args + ["--cuda_version", args.cuda_version]
            if args.cuda_home:
                build_args = build_args + ["--cuda_home", args.cuda_home]
            if args.cudnn_home:
                build_args = build_args + ["--cudnn_home", args.cudnn_home]

    build_onnxruntime(args.onnxruntime_home, args.config, build_args, "all_eps", args)

