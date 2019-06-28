# Perf-test Image

This image is for automate the process of performance tuning for onnxruntime. Given a model you'd like to optimize its performance, it will iterate through all available execution providers, environment variables, and run option combinations. The outputs of the image are a .json file which summarizes the latency results for all combinations the image has searched, and profiling files for the top 5 combinations.

Currently the execution provider available to tune are cpu, cpu_openmp, mkldnn, mkldnn_openmp, mkldnn_mklml, cuda, tensorrt, and ngraph.  

## Build the Image Locally

To build `perf-test` image locally, you must first build `onnxruntime` using `build_perf_test.py` to create builds for different execution providers on Linux. 

### Prerequisites:
- Python 3.7+ `sudo apt install python3.7`
- CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
- CuDNN https://developer.nvidia.com/cudnn 
- TensorRT https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-debian
- ONNX Runtime https://github.com/Microsoft/onnxruntime

### Example build command:  
```
python3 build_perf_test.py --onnxruntime_home <path_to_onnxruntime> --use_cuda --cuda_home <path_to_cuda> --cudnn_home <path_to_cudnn> --use_tensorrt --tensorrt_home <path_to_tensorrt> --use_ngraph
```

### build_perf_test.py args

`--onnxruntime_home`:   Required. Your local ONNX Runtime source directory. 

`--use_cuda`: Flag to build ONNX runtime with CUDA execution provider. Required if you'd like to tune performace with CUDA execution provider and GPU is available on your machine.

`--cuda_home`: The path to your cuda installation. e.g. /usr/local/cuda . Required if --use_cuda is used. 

`--cuda_version`: Optional. The version of CUDA toolkit to use. Auto-detect if not specified. e.g. 9.0

`--cudnn_home`: The path to your cuDNN installation which contains libcudnn.so* files. Required if --use_cuda is used. 

`--use_tensorrt`: Flag to build ONNX runtime with TensorRT execution provider. Required if you'd like to tune performace with TensorRT execution provider and GPU is available on your machine.

`--tensorrt_home`: The path to your TensorRT installation, which contains lib/libnvinfer.so*. Required if --use_tensorrt is used. 

`--use_ngraph`: Flag to build ONNX runtime with nGraph execution provider. Required if you'd like to tune performance with nGraph. 

`--use_mklml`: Flag to build ONNX runtime with MKLML execution provider. Required if you'd like to tune performance with MKLML. 

`--variants`: Optional. Specify execution providers to build. Each execution provider is separated by ",". For example, `--variants cpu,cuda`. Available options are cpu, cpu_openmp, mkldnn, mkldnn_openmp, mkldnn_mklml, cuda, tensorrt, ngraph. If not specified, build all. 

`--config`: Optional. ONNX runtime build configuration. Available options are "Debug", "MinSizeRel", "Release", "RelWithDebInfo". Default is "RelWithDebInfo". 

## Build and Run the Image
To build Docker container:  
`docker build -t perf_test -f Dockerfile.perftest .`

To run Docker perf_test Image:  
`docker run perf_test <path_to_onnx_model> <path_to_results_file>`


### Windows

Prerequisites:
- Python 3.7+
- CUDA https://developer.nvidia.com/cuda-toolkit
- cuDNN https://developer.nvidia.com/cudnn
- TensorRT https://developer.nvidia.com/tensorrt
- ONNX Runtime https://github.com/Microsoft/onnxruntime

After required libraries are installed you can build onnxruntime using the command:  
`
python build_perf_test.py --onnxruntime_home D:\depot\onnxruntime --use_cuda --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0" --cudnn_home D:\tmp\cudnn-10.0-windows10-x64-v7.5.0.56\cuda --use_tensorrt --tensorrt_home D:\tmp\TensorRT-5.1.2.2.Windows10.x86_64.cuda-10.0.cudnn7.5\TensorRT-5.1.2.2
`

This will build several versions of onnxruntime and put binaries to bin folder. To rebuild a specific version use `--variants` parameter, e.g. `--variants cpu,cuda`

Now you can run perf_test using command `python perf_test.py <path_to_onnx_model> <path_to_results_file>`. You can use the same arguments as for onnxruntime_pert_test tool, e.g. -m for mode, -e to specify execution provider etc. By default it will try all providers available.

### Linux
