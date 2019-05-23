# pert_test tool

## Windows

Prerequisites:
- Python 3.7+
- CUDA https://developer.nvidia.com/cuda-toolkit
- cuDNN https://developer.nvidia.com/cudnn
- TensorRT https://developer.nvidia.com/tensorrt

After required libraries are installed you can build onnxruntime using the command:  
`
python build_perf_test.py --onnxruntime_home D:\depot\onnxruntime --use_cuda --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0" --cudnn_home D:\tmp\cudnn-10.0-windows10-x64-v7.5.0.56\cuda --use_tensorrt --tensorrt_home D:\tmp\TensorRT-5.1.2.2.Windows10.x86_64.cuda-10.0.cudnn7.5\TensorRT-5.1.2.2
`

This will build several versions of onnxruntime and put binaries to bin folder. To rebuild a specific version use `--variants` parameter, e.g. `--variants cpu,cuda`

Now you can run perf_test using command `python perf_test.py <path_to_onnx_model> <path_to_results_file>`. You can use the same arguments as for onnxruntime_pert_test tool, e.g. -m for mode, -e to specify execution provider etc. By default it will try all providers available.

## Linux

Prerequisites:
- Python 3.7+ `sudo apt install python3.7`
- CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
- TensorRT https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-debian

Build command:  
`python3 build_perf_test.py --onnxruntime_home /home/artr/repo/onnxruntime --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr --use_tensorrt --tensorrt_home /usr --use_ngraph`

To build Docker container:  
`docker build -t perf_test -f Dockerfile.perftest .`

To run perf_test:  
`python3.7 perf_test.py <path_to_onnx_model> <path_to_results_file>`
