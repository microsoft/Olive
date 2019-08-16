# Performance Tuning Image

This image is for automating the process of performance tuning in ONNX Runtime. Given an ONNX model you'd like to optimize the performance, the image will strategically search through all combinations of available execution providers, environment variables, and run options. Finally it outputs a JSON file that summarizes the latency results for the best combinations the image has searched, and profiling files for the combinations with top performance for each execution provider.

Currently the execution providers available are cpu, cpu_openmp, mkldnn, mkldnn_openmp, mklml, cuda, tensorrt, and ngraph.  

To use the image, you can either [pull from Azure Registry](#Pull-and-Run-the-Image-From-Azure-Registry) or [build and run locally](#Build-and-Run-the-Image-Locally) from this repo.

## Pull and Run the Image From Azure Registry

A pre-built version of the image is available at Azure Registry. Once you have docker installed, you can easily pull and run the image on Linux as well as on Windows. 

With the correct credentials, you can pull the image directly using 
```
docker pull ziylregistry.azurecr.io/perf-test
```

Upon success, run Docker perf-test image by
```
docker run [--runtime=nvidia] ziylregistry.azurecr.io/perf-test --model <path_to_onnx_model> --result <path_to_result_dir> [other optional args]
```
or 
```
docker run [--runtime=nvidia] ziylregistry.azurecr.io/perf-test --input_json <input_json_file>
```

### Perf-test Image Arguments

`--model`: Requried or specify in --input_json. The ONNX model to perform performance tuning. 

`--result`: Required or specify in --input_json. The directory to put output files. 

`--config`: ONNX Runtime configuration. Available options are "Debug", "MinSizeRel", "Release", "RelWithDebInfo". Default is "RelWithDebInfo". 

`--mode`: Specifies the test mode. Value could be "duration" or "time".

`--execution_provider`: Execution Provider. Available options are "cpu", "cpu_openmp", "cuda", "tensorrt", "ngraph", "mkldnn", "mkldnn_openmp", and "mkldnn_mklml"

`--repeated_times`: The repeated times if running in 'times' test mode. Default:20.

`--duration_times`: The seconds to run for 'duration' mode. Default:10.

`--threadpool_size`: Threadpool size if parallel executor (--parallel) is enabled. Default is the number of cores. 

`--num_threads`: OMP_NUM_THREADS value. Default is the number of cores. 

`--top_n`: Show percentiles for top n runs in each execution provider. Default:3.

`--parallel`: Tune performance using parallel executor. Default is True. 

`--optimization_level`: Default=3. 0: disable optimization, 1: basic optimization, 2: extended optimization, 3: extended+layout optimization.

`--input_json`: A JSON file specifying the run specs above. For example, 
```
{
    "model": "resnet50/model.onnx",
    "result": "output",
    "mode": "times", 
    "config": "RelWithDebInfo", 
    "execution_provider": "",
    "repeated_times": "20",
    "duration_time": "10",
    "threadpool_size": "",
    "num_threads": "5",
    "top_n": "5",
    "parallel": "True"
}
```

## Build and Run the Image Locally

Alternatively, you can build and run perf-test image locally by following the steps below. 

### 1. Build ONNX Runtime
To use `perf-test` locally, you must first build `onnxruntime` using `build_perf_test.py` to create builds for different execution providers. If you prefer to build `perf-test` with docker, then you need to run `build_perf_test.py` on Linux.  

#### Prerequisites:
- Python 3.7+ `sudo apt install python3.7`
- ONNX Runtime https://github.com/Microsoft/onnxruntime

- CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
- CuDNN https://developer.nvidia.com/cudnn 
- TensorRT https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-debian

#### Example build command:  
```
python3.7 build_perf_test.py \
 --onnxruntime_home <path_to_onnxruntime> \
 --use_cuda --cuda_home <path_to_cuda> --cudnn_home <path_to_cudnn> \
 --use_tensorrt --tensorrt_home <path_to_tensorrt> \
 --use_ngraph \
 --use_mklml
```

#### build_perf_test.py args

`--onnxruntime_home`:   Required. Your local ONNX Runtime source directory. 

`--use_cuda`: Flag to build ONNX runtime with CUDA execution provider. Required to tune performace with CUDA execution provider. A GPU must be available on your machine.

`--cuda_home`: The path to your cuda installation. e.g. /usr/local/cuda . Required if --use_cuda is used. 

`--cuda_version`: Optional. The version of CUDA toolkit to use. Auto-detect if not specified. e.g. 9.0

`--cudnn_home`: The path to your CuDNN installation. The path should  contain libcudnn.so* files if on Linux, or contiain bin/cudnn*.dll files if on Windows. Required if --use_cuda is used. 

`--use_tensorrt`: Flag to build ONNX runtime with TensorRT execution provider. Required to tune performace with TensorRT execution provider. A GPU must be available on your machine.

`--tensorrt_home`: The path to your TensorRT installation. The path should contain lib/libnvinfer.so* if on Linux, or contain lib/nvinfer.dll if on Windows. Required if --use_tensorrt is used. 

`--use_ngraph`: Flag to build ONNX runtime with nGraph execution provider. Required to tune performance with nGraph. 

`--use_mklml`: Flag to build ONNX runtime with MKLML execution provider. Required to tune performance with MKLML. 

`--variants`: Optional. Specify execution providers to build. Each execution provider is separated by ",". For example, `--variants cpu,cuda`. Available options are cpu, cpu_openmp, mkldnn, mkldnn_openmp, mkldnn_mklml, cuda, tensorrt, and nraph. If not specified, build all. 

`--config`: Optional. ONNX runtime build configuration. Available options are "Debug", "MinSizeRel", "Release", "RelWithDebInfo". Default is "RelWithDebInfo". 

### 2. Build Docker Image
If an ONNX Runtime Linux build is completed in step 1, you can build the image with docker by running 
```
docker build -t perf-test -f Dockerfile.perftest .
```

If ONNX Runtime is built on Windows, jump to [Run perf-test Without Docker](#4-run-perf-test-without-docker)

### 3. Run Docker Image
```
docker run [--runtime=nvidia] perf-test --model <path_to_onnx_model> --result <path_to_result_dir> [other optional args]
```
or
```
docker run [--runtime=nvidia] perf-test --input_json <input_json_file>
```

### 4. Run `perf-test` Without Docker

You can choose to run `perf_test.py` locally if docker is not available or ONNX Runtime is built on Windows. 

You can run perf_test using command 
```
python perf_test.py --model <path_to_onnx_model> --result <path_to_results_dir> [other optional args]
```
The optional arguments are the same as for perf-test images. By default it will try all providers available.
