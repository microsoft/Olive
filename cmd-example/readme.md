# OLive Command Line Tool

This repository shows how to deploy and use OLive by running commands.

## Prerequisites
Download OLive pakcage [here](https://olivewheels.blob.core.windows.net/repo/onnxruntime_olive-0.2.0-py3-none-any.whl) and install with command `pip install onnxruntime_olive-0.2.0-py3-none-any.whl`

ONNX Runtime pakcage can be installed with

`pip install --extra-index-url https://olivewheels.azureedge.net/test onnxruntime_openvino_dnnl==1.9.0` for cpu

or 

`pip install --extra-index-url https://olivewheels.azureedge.net/test onnxruntime_gpu_tensorrt==1.9.0` for gpu

## How to use
User can call `olive convert` or `olive optimize` with related arguments. OLive will manage python package installation, such as `ONNX Runtime` or `TensorFlow` or `PyTorch`. 

User can also run `olive convert` or `olive optimize` with `use_docker` or `use_conda` options, if `docker` or `conda` is installed. In this way, OLive will run service in container or in a new conda environment.

For `olive optimize`, user can set `--use_gpu` to run optimization service with ONNX Runtime gpu package. 

### For Model Framework Conversion
Converts models from PyTorch and TensorFlow model frameworks to ONNX, and tests the converted models' correctness.

Here are arguments for OLive Conversion:

| Argument | Detail | Example |
|:--|:--|:--|
| **model_path** | (required) model path for conversion | test.pb |
|**model_framework**| (required) model original framework | tensorflow |
|**model_root_path**| (optional) model path for conversion, only for PyTorch model | D:\model\src |
|**input_names**| (optional) comma-separated list of names of input nodes of model | title_lengths:0,title_encoder:0,ratings:0 |
|**output_names**| (optional) comma-separated list of names of output nodes of model | output_identity:0,loss_identity:0 |
|**input_shapes**| (optional) list of shapes of each input node. The order of the input shapes should be the same as the order of input names | [[1,7],[1,7],[1,7]] |
|**output_shapes**| (optional) list of shapes of each output node. The order of the output shapes should be the same as the order of output names | [[1,64],[1,64]] |
|**input_types**| (optional) comma-separated list of types of input nodes. The order of the input types should be the same as the order of input names | float32,float32,float32 |
|**output_types**| (optional) comma-separated list of types of output nodes. The order of the output types should be the same as the order of output names | float32,float32 |
|**onnx_opset**| (optional) target opset version for conversion | 11 |
|**onnx_model_path**| (optional) ONNX model path as conversion output | test.onnx |
|**sample_input_data_path**| (optional) path to sample input data | sample_input_data.npz |
|**framework_version**| (optional) original framework version | 1.14 |

There are two ways to call OLive convert with cmd. 

Test model and sample input data can be downloaded here: [model](https://olivemodels.blob.core.windows.net/models/conversion/full_doran_frozen.pb), [sample test data ](https://olivemodels.blob.core.windows.net/models/conversion/doran.npz).

1. With all inline arguments. For example:
    ```
    olive convert 
    --model_path full_doran_frozen.pb 
    --model_framework tensorflow 
    --framework_version 1.13 
    --input_names title_lengths:0,title_encoder:0,ratings:0,query_lengths:0,passage_lengths:0,features:0,encoder:0,decoder:0,Placeholder:0 
    --output_names output_identity:0,loss_identity:0 
    --sample_input_data_path doran.npz
    ```

2. With conversion config file with all needed arguments: 
    ```
    {
      "model_path": "full_doran_frozen.pb",
      "input_names": ["title_lengths:0", "title_encoder:0", "ratings:0", "query_lengths:0", "passage_lengths:0", "features:0", "encoder:0", "decoder:0", "Placeholder:0"],
      "output_names": ["output_identity:0", "loss_identity:0"],
      "sample_input_data_path": "doran.npz"
    }
    ```
    `olive convert --conversion_config cvt_config.json --model_framework tensorflow --framework_version 1.13` where cvt_config.json file includes related configurations. 

Then the ONNX model will be saved in `onnx_model_path`, by default `res.onnx`

### For ONNX Model Inference Optimization
Tunes different execution providers, inference session options, and environment variable options for the ONNX model with ONNX Runtime. Selects and outputs the option combinations with the best performance for lantency or throughput.

Here are arguments for OLive optimization:
| Argument  | Detail  | Example |
|--|--|--|
| **model_path** | (required) model path for optimization | PytorchBertSquad.onnx |
| **throughput_tuning_enabled** | (optional)whether tune model for optimal throughput |  |
| **max_latency_percentile** | (required for throughput tuning) throughput max latency pct tile | 0.95 |
| **max_latency_sec** | (required for throughput tuning) max latency in pct tile in second | 0.05 |
| **dynamic_batching_size** | (specified for throughput tuning) max batchsize for dynamic batching | 1 |
| **threads_num** | (specified for throughput tuning) threads num for throughput optimization | 4 |
| **min_duration_sec** | (specified for throughput tuning) 	minimum duration for each run in second | 10 |
| **result_path** | (optional) result directory for OLive optimization | olive_opt_result |
| **input_names** | (optional) comma-separated list of names of input nodes of model. Required for ONNX model with dynamic inputs' size | input_ids,input_mask,segment_ids |
| **output_names** | (optional) comma-separated list of names of output nodes of model | scores |
| **input_shapes** | (optional) list of shapes of each input node. The order of the input shapes should be the same as the order of input names. Required for ONNX model with dynamic inputs' size | [[1,7],[1,7],[1,7]] |
| **providers_list** | (optional) providers used for perftuning | cpu,dnnl |
| **trt_fp16_enabled** | (optional) whether enable fp16 mode for TensorRT |  |
| **quantization_enabled** | (optional) whether enable quantization optimization or not |  |
| **transformer_enabled** | (optional) whether enable transformer optimization or not |  |
| **transformer_args** | (optional) onnxruntime transformer optimizer args | "--model_type bert" |
| **sample_input_data_path** | (optional) path to sample input data | sample_input_data.npz |
| **concurrency_num** | (optional) tuning process concurrency number | 2 |
| **kmp_affinity** | (optional) bind OpenMP* threads to physical processing units | respect,none |
| **omp_max_active_levels** | (optional) maximum number of nested active parallel regions | 1 |
| **inter_thread_num_list** | (optional) list of inter thread number for perftuning | 1,2,4 |
| **intra_thread_num_list** | (optional) list of intra thread number for perftuning | 1,2,4 |
| **execution_mode_list** | (optional) list of execution mode for perftuning | parallel,sequential |
| **ort_opt_level_list** | (optional) onnxruntime optimization level | all |
| **omp_wait_policy_list** | (optional) list of OpenMP wait policy for perftuning | active |
| **warmup_num** | (optional) warmup times for latency measurement | 20 |
| **test_num** | (optional) repeat test times for latency measurement | 200 |

There are two ways to call OLive optimize with cmd.

Test model can be downloaded [here](https://olivemodels.blob.core.windows.net/models/optimization/TFBertForQuestionAnswering.onnx).

To optimize ONNX model latency:

1. With all inline arguments. For example:
    ```
    olive optimize 
    --model_path TFBertForQuestionAnswering.onnx 
    --input_names attention_mask,input_ids,token_type_ids 
    --input_shapes [[1,7],[1,7],[1,7]] 
    --quantization_enabled
    ```

2. With optimization config file with all needed arguments: 
    ```
    {
        "model_path": "TFBertForQuestionAnswering.onnx",
        "input_names": ["attention_mask", "input_ids", "token_type_ids"],
        "input_shapes": [[1,7],[1,7],[1,7]],
        "quantization_enabled": true,
        "providers_list": ["cpu", "dnnl"],
        "omp_max_active_levels": ["1"],
        "kmp_affinity": ["respect,none"],
        "concurrency_num": 2
    }
    ```
    `olive optimize --optimization_config opt_config.json` where opt_config.json file includes related configurations. 

To optimize ONNX model throughput:

1. With all inline arguments. For example:
    ```
    olive optimize 
    --model_path TFBertForQuestionAnswering.onnx 
    --input_names attention_mask,input_ids,token_type_ids 
    --input_shapes [[-1,7],[-1,7],[-1,7]] 
    --throughput_tuning_enabled 
    --max_latency_percentile 0.95 
    --max_latency_sec 0.1 
    --threads_num 1 
    --dynamic_batching_size 4 
    --min_duration_sec 10
    ```

2. With optimization config file with all needed arguments: 
    ```
    {
        "model_path": "TFBertForQuestionAnswering.onnx",
        "input_names": ["attention_mask", "input_ids", "token_type_ids"],
        "input_shapes": [[-1,7],[-1,7],[-1,7]],
        "throughput_tuning_enabled": true,
        "max_latency_percentile": 0.95,
        "max_latency_sec": 0.1,
        "threads_num": 1,
        "dynamic_batching_size": 4,
        "min_duration_sec": 10
    }
    ```
    `olive optimize --optimization_config opt_config.json` where opt_config.json file includes related configurations. 

Then the result JSON file and optimized model will be stored in `result_path`, by default `olive_opt_result`
