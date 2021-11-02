# OLive - ONNX Runtime Go Live
OLive, meaning ONNX Runtime(ORT) Go Live, is a python package that automates the process of accelerating models with [ONNX Runtime(ORT)](https://onnxruntime.ai/). It contains two parts including model conversion to [ONNX](https://onnx.ai/) with correctness checking and auto performance tuning with ORT. Users can run these two together through a single pipeline or run them independently as needed.
### Model conversion to ONNX
Simplify multiple frameworks to ONNX conversion experience by integrating existing [ONNX conversion tools](https://github.com/onnx/tutorials#converting-to-onnx-format) into a single package, as well as validating the converted models' correctness. Currently supported frameworks are PyTorch and TensorFlow.
 * TensorFlow: OLive supports conversion with TensorFlow model in saved model, frozen graph, and checkpoint format. User needs to provider inputs' names and outputs' names for frozen graph and checkpoint conversion.
 * PyTorch: User needs to provide inputs' names and shapes to convert PyTorch model. Besides, user needs to provide outputs' names and shapes to convert torchscript PyTorch model.

### Auto performance tuning with ORT
ONNX Runtime(ORT) is a high performance inference engine to run ONNX model. It enables many advanced tuning knobs for user to further optimize inference performance. OLive heuristically explores optimization search space in ORT to select the best ORT settings for a specific model on a specific hardware.  It outputs the option combinations with the best performance.

User needs to provide inputs' names and shapes for ONNX model with dynamic inputs' size. 

Optimization fileds:
* [Execution Providers](https://onnxruntime.ai/docs/execution-providers/):
   * MLAS(default CPU EP), Intel DNNL and OpenVino for CPU
   * Nvidia CUDA and TensorRT for GPU
* Environment Variables:
   * OMP_WAIT_POLICY: 
   * OMP_NUM_THREADS
   * KMP_AFFINITY
   * OMP_MAX_ACTIVE_LEVELS
* [Session Options](https://onnxruntime.ai/docs/performance/tune-performance.html#default-cpu-execution-provider-mlas):
   * inter_op_num_threads
   * intra_op_num_threads
   * execution_mode
   * graph_optimization_level
 * [INT8 Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
 * [Transformer Model Optimization](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers)

## Getting Started
OLive package can be downloaded [here](https://olivewheels.blob.core.windows.net/repo/onnxruntime_olive-0.1.0-py3-none-any.whl) and installed with command `pip install onnxruntime_olive-0.1.0-py3-none-any.whl`

There are three ways to use OLive:
1. [Use With Command Line](./cmd-example/readme.md): Run the OLive with command line using Python. 
2. [Use With Jupyter Notebook](./notebook-tutorial): Quickstart of the OLive with tutorial using Jupyter Notebook. 
3. [Use With OLive Server](./server-example/readme.md): Setup local OLive server for model conversion, optimizaton, and visualization service.

## Inference your model with OLive result from auto performance tuning 
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
10/28/2021

Update OLive from docker container based usage to python package based usage for more flexibilities.

Enable more optimization options for performance tuning with ORT, including INT8 quantization, mix precision in ORT-TensorRT, and transformer model optimization.

## Contributing
We’d love to embrace your contribution to OLive. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
   
   
