# OLive - ONNX Runtime Go Live
OLive, meaning ONNX Runtime(ORT) Go Live, is a python package that simplifies the process of accelerating models with [ONNX Runtime(ORT)](https://github.com/microsoft/onnxruntime). It contains two parts of model conversion to ONNX with correctness checking and auto performance tuning with ORT. Users can run these two together through a single pipelie or run them independently as needed.
* Model conversion to ONNX: to output the converted ONNX model
* Auto performance tuning with ORT: to output the optimized ONNX model and a file of the tuned inference latency under corresponding ORT settings 

## Getting Started
OLive package can be downloaded [here](https://olivewheels.blob.core.windows.net/repo/onnxruntime_olive-0.1.0-py3-none-any.whl) and installed with command `pip install onnxruntime_olive-0.1.0-py3-none-any.whl`

There are three ways to use OLive:
1. [Use With Command Line](./cmd-example/readme.md): Run the OLive with command line using Python. 
2. [Use With Jupyter Notebook](./notebook-tutorial): Quickstart of the OLive with tutorial using Jupyter Notebook. 
3. [Use With OLive Server](./server-example/readme.md): Setup local OLive server for model conversion, optimizaton, and visualization service.

## Inference your mdoel with OLive result from Auto performance tuning 
1. Get best tuning result with `best_test_name`, which includes inference session settings, environment variable settings, and latency result. 
2. Set related environment variables in your environment.
    * OMP_WAIT_POLICY
    * OMP_NUM_THREADS
    * KMP_AFFINITY
    * OMP_MAX_ACTIVE_LEVELS
    * ORT_TENSORRT_FP16_ENABLE
3. Create onnxruntime inference session with related settings.
    * inter_op_num_threads
    * intra_op_num_threads
    * execution_mode
    * graph_optimization_level
    * execution_provider
    ```
   import onnxruntime as ort
   sess_options = ort.SessionOptions()
   sess_options.inter_op_num_threads = inter_op_num_threads
   sess_options.intra_op_num_threads = intra_op_num_threads
   sess_options.execution_mode = execution_mode
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
   onnx_session = ort.InferenceSession(model_path, sess_options, providers=[execution_provider])
    ```

## Key Updates
OLive service changes from docker container based into python package based, which gives user more flexibilities. 

Inference optimization options updates. Now OLive supports more optimization options, for example INT8 quantization, TensorRT FP16, and transformer model optimization. 

## Contributing
We’d love to embrace your contribution to OLive. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
   
   
