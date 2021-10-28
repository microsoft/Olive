# OLive - ONNX Go Live
OLive, meaning ONNX Go Live, is a python package that optimizes the process of ONNX model shipping. It integrates model conversion including correctness test, and performance optimization into a single pipeline.

# Getting Started
OLive package can be downloaded [here](https://olivewheels.blob.core.windows.net/repo/onnxruntime_olive-0.1.0-py3-none-any.whl) and installed with command `pip install onnxruntime_olive-0.1.0-py3-none-any.whl`

There are three ways to use OLive:
1. [Use With Command Line](./cmd-example/readme.md): Run the OLive with command line using Python. 
2. [Use With Jupyter Notebook](./notebook-tutorial): Quickstart of the OLive with tutorial using Jupyter Notebook. 
3. [Use With OLive Server](./server-example/readme.md): Setup local OLive server for model conversion, optimizaton, and visualization service.

# Inference with OLive result
1. Get best tuning result with `best_test_name`, which includes inference session settings, environment variable settings, and latency result. 
2. Set related environment variables.
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
   
   