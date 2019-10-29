# Performance Tuning Image

This image is for automating the process of performance tuning in ONNX Runtime. Given an ONNX model you'd like to optimize the performance, the image will strategically search through all combinations of available execution providers, environment variables, and run options. Finally it outputs a JSON file that summarizes the latency results for the best combinations the image has searched, and profiling files for the combinations with top performance for each execution provider.

Currently the execution providers available are cpu, cpu_openmp, mkldnn, mklml, cuda, tensorrt, and ngraph.  

To use the image, you can either [pull from Microsoft Container Registry](#Pull-and-Run-the-Image-From-Microsoft-Container-Registry) or [build and run locally](#Build-and-Run-the-Image-Locally) from this repo.

## Pull and Run the Image From Microsoft Container Registry

A pre-built version of the image is available at Microsoft Container Registry. Once you have docker installed, you can easily pull and run the image on Linux as well as on Windows. 

With the correct credentials, you can pull the image directly using 
```
docker pull mcr.microsoft.com/onnxruntime/perf-tuning
```

Upon success, run Docker perf-tuning image by
```
docker run [--runtime=nvidia] mcr.microsoft.com/onnxruntime/perf-tuning --model <path_to_onnx_model> --result <path_to_result_dir> [other optional args]
```
or 
```
docker run [--runtime=nvidia] mcr.microsoft.com/onnxruntime/perf-tuning --input_json <input_json_file>
```

### perf-tuning Image Arguments

`--model`: Requried or specify in --input_json. The ONNX model to perform performance tuning. 

`--result`: Required or specify in --input_json. The directory to put output files. 

`--config`: ONNX Runtime configuration. Available options are "Debug", "MinSizeRel", "Release", "RelWithDebInfo". Default is "RelWithDebInfo". 

`--test_mode`: Specifies the test mode. Value could be "duration" or "time". Default is "time".

`--execution_provider`: Execution Provider. Available options are "cpu", "cpu_openmp", "cuda", "tensorrt", "ngraph", "mkldnn", and "mklml"

`--repeated_times`: The repeated times if running in 'times' test mode. Default:20.

`--duration_times`: The seconds to run for 'duration' mode. Default:10.

`--intra_op_num_threads`: Sets the number of threads used to parallelize the execution within nodes. A value of 0 means the test will auto-select a default. Must >=0. 

`--num_threads`: OMP_NUM_THREADS value. Default is the number of cores. 

`--top_n`: Show percentiles for top n runs in each execution provider. Default:3.

`--parallel`: Tune performance using parallel executor. Default is True. 

`--optimization_level`: Default=3. 0: disable optimization, 1: basic optimization, 2: extended optimization, 3: extended+layout optimization.

`--input_json`: A JSON file specifying the run specs above. For example, 
```
{
    "model": "resnet50/model.onnx",
    "result": "output",
    "test_mode": "times", 
    "config": "RelWithDebInfo", 
    "execution_provider": "",
    "repeated_times": "20",
    "duration_time": "10",
    "intra_op_num_threads": "",
    "num_threads": "5",
    "top_n": "5",
    "parallel": "True"
}
```

## Build and Run the Image Locally

You can also build and run perf-tuning image based on your local ONNX Runtime locally by following the steps below. 

### 1. Build ONNX Runtime
To use `perf-tuning` locally, you must first build `onnxruntime` using `build_perf_tuning.py` to create builds for different execution providers. If you prefer to build `perf-tuning` with docker, then you need to run `build_perf_tuning.py` on Linux.  

#### Prerequisites:
- Python 3.7+ `sudo apt install python3.7`
- ONNX Runtime https://github.com/Microsoft/onnxruntime

- CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
- CuDNN https://developer.nvidia.com/cudnn 
- TensorRT https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-debian

#### Example build command:  
```
python3.7 build_perf_tuning.py \
 --onnxruntime_home <path_to_onnxruntime> \
 --use_cuda --cuda_home <path_to_cuda> --cudnn_home <path_to_cudnn> \
 --use_tensorrt --tensorrt_home <path_to_tensorrt> \
 --use_ngraph \
 --use_mklml
```

#### build_perf_tuning.py args

`--onnxruntime_home`:   Required. Your local ONNX Runtime source directory. 

`--use_cuda`: Flag to build ONNX Runtime with CUDA execution provider. Required to tune performace with CUDA execution provider. A GPU must be available on your machine.

`--cuda_home`: The path to your cuda installation. e.g. /usr/local/cuda . Required if --use_cuda is used. 

`--cuda_version`: Optional. The version of CUDA toolkit to use. Auto-detect if not specified. e.g. 9.0

`--cudnn_home`: The path to your CuDNN installation. The path should  contain libcudnn.so* files if on Linux, or contiain bin/cudnn*.dll files if on Windows. Required if --use_cuda is used. 

`--use_tensorrt`: Flag to build ONNX Runtime with TensorRT execution provider. Required to tune performace with TensorRT execution provider. A GPU must be available on your machine.

`--tensorrt_home`: The path to your TensorRT installation. The path should contain lib/libnvinfer.so* if on Linux, or contain lib/nvinfer.dll if on Windows. Required if --use_tensorrt is used. 

`--use_ngraph`: Flag to build ONNX Runtime with nGraph execution provider. Required to tune performance with nGraph. 

`--use_mklml`: Flag to build ONNX Runtime with MKLML execution provider. Required to tune performance with MKLML. 

`--variants`: Optional. Specify execution providers to build. Each execution provider is separated by ",". For example, `--variants cpu,cuda`. Available options are cpu, cpu_openmp, mkldnn, mklml, cuda, tensorrt, and nraph. If not specified, build all. 

`--config`: Optional. ONNX Runtime build configuration. Available options are "Debug", "MinSizeRel", "Release", "RelWithDebInfo". Default is "RelWithDebInfo". 

### 2. Build Docker Image
If an ONNX Runtime Linux build is completed in step 1, you can build the image with docker by running 
```
docker build -t perf-tuning .
```

If ONNX Runtime is built on Windows, jump to [Run perf-tuning Without Docker](#4-run-perf-tuning-without-docker)

### 3. Run Docker Image
```
docker run [--runtime=nvidia] perf-tuning --model <path_to_onnx_model> --result <path_to_result_dir> [other optional args]
```
or
```
docker run [--runtime=nvidia] perf-tuning --input_json <input_json_file>
```

### 4. Run `perf-tuning` Without Docker

You can choose to run `perf_tuning.py` locally if docker is not available or ONNX Runtime is built on Windows. 

You can run perf_tuning using command 
```
python perf_tuning.py --model <path_to_onnx_model> --result <path_to_results_dir> [other optional args]
```
The optional arguments are the same as for perf-tuning images. By default it will try all providers available.
