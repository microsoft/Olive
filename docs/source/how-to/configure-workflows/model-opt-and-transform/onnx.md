# ONNX

[ONNX](https://onnx.ai/) is an open graph format to represent machine learning models. [ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.

Olive provides multiple transformations and optimizations based on various ONNX to improve model performance.

## Model Optimizer
`OnnxPeepholeOptimizer` optimizes an ONNX model by fusing nodes. Fusing nodes involves merging multiple nodes in a model into a single node to
reduce the computational cost and improve the performance of the model. The optimization process involves analyzing the structure of the ONNX model and identifying nodes that can be fused.

Also, inserts a `Cast` operation for cases where `ArgMax` input. For example, before ONNXRuntime 1.20, TensorProto.INT64 isn't supported on CPU or CUDA EP so a `Cast` operator inserted to cast the inputs to TensorProto.INT32.

Please refer to [OnnxPeepholeOptimizer](../../../reference/pass.rst#onnx_peephole_optimizer) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OnnxPeepholeOptimizer"
}
```

## ORT Transformers Optimization
While ONNX Runtime automatically applies most optimizations while loading transformer models, some of the latest optimizations that have not
yet been integrated into ONNX Runtime.
`OrtTransformersOptimization` provides an offline capability to optimize [transformers](https://huggingface.co/docs/transformers/index) models
in scenarios where ONNX Runtime does not apply the optimization at load time.
These optimizations are provided by onnxruntime through
[onnxruntime.transformers](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers). Please
refer to the [corresponding documentation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md)
for more details on the optimizations done by this tool.

Please refer to [OrtTransformersOptimization](../../../reference/pass.rst#ort_transformers_optimization) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OrtTransformersOptimization",
    "model_type": "bert"
}
```
## Append Pre/Post Processing Ops
`AppendPrePostProcessingOps` inserts pre and post processing ops into the ONNX graph.

### Example Configuration
```json
{
    "type": "AppendPrePostProcessingOps",
    "tool_command": "superresolution",
    "tool_command_args": {
        "output_format": "png"
    }
}
```
```json
{
    "type": "AppendPrePostProcessingOps",
    "tool_command": "whisper",
    "tool_command_args": {
        "use_audio_decoder": true
    }
}
```

`AppendPrePostProcessingOps` also supports pre/post processing ops by leveraging the [onnxruntime-extension steps](https://github.com/microsoft/onnxruntime-extensions/tree/main/onnxruntime_extensions/tools/pre_post_processing/steps) and `PrePostProcessor`.
You can refer to [here](https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/Example%20usage%20of%20the%20PrePostProcessor.md) to see how to leverage `PrePostProcessor` to customize pre and post processing ops.

* Olive introduces two placeholders to represent the model input/output shape dimension value: `__model_input__` and `__model_output__`.
* To support the IoMapEntry, the step need choose use the full form. For example:
```json
    "YCbCrToPixels": {
        "params": {
            "layout": "BGR",
        },
        "io_map": [
            ["Y1_uint8", 0, 0],
            ["Cb1_uint8", 0, 1],
            ["Cr1_uint8", 0, 2],
        ],
    }
```
* The `tool_command_args` will be used to describe the input parameters to create the `PrePostProcessor` instance. It is list of `PrePostProcessorInput`.
  The `name` is the tensor name. The `data_type` and `shape` will be used to create the tensor type. The `shape` can be a list of integers or a list of string.

Users that write their own pre/post processing steps need to have the knowledge about whether the step includes the operators that is built-in support or supported in onnxruntime-extensions.
For example, for some ops like `ConvertImageToBGR` which requires other extensions may be incompatible with ort-web, user need to exclude this kind of ops to generate proper models.

Here are some examples to describe the pre/post processing which is exactly same with [superresolution](https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/add_pre_post_processing_to_model.py#L89)

```json
{
    "pre": [
        {"ConvertImageToBGR": {}},
        {
            "Resize": {
                "resize_to": [
                    {"type": "__model_input__", "input_index": 0, "dim_index": -2},
                    {"type": "__model_input__", "input_index": 0, "dim_index": -1},
                ]
            }
        },
        {
            "CenterCrop": {
                "height": {"type": "__model_input__", "input_index": 0, "dim_index": -2},
                "width": {"type": "__model_input__", "input_index": 0, "dim_index": -1},
            }
        },
        {"PixelsToYCbCr": {"layout": "BGR"}},
        {"ImageBytesToFloat": {}},
        {"Unsqueeze": {"axes": [0, 1]}},
    ],
    "post": [
        {"Squeeze": {"axes": [0, 1]}},
        {"FloatToImageBytes": {"name": "Y1_uint8"}},
        {
            "Resize": {
                "params": {
                    "resize_to": [
                        {"type": "__model_output__", "output_index": 0, "dim_index": -2},
                        {"type": "__model_output__", "output_index": 0, "dim_index": -1},
                    ],
                    "layout": "HW",
                },
                "io_map": [["PixelsToYCbCr", 1, 0]],
            }
        },
        {"FloatToImageBytes": {"multiplier": 1.0, "name": "Cb1_uint8"}},
        {
            "Resize": {
                "params": {
                    "resize_to": [
                        {"type": "__model_output__", "output_index": 0, "dim_index": -2},
                        {"type": "__model_output__", "output_index": 0, "dim_index": -1},
                    ],
                    "layout": "HW",
                },
                "io_map": [["PixelsToYCbCr", 2, 0]],
            }
        },
        {"FloatToImageBytes": {"multiplier": 1.0, "name": "Cr1_uint8"}},
        {
            "YCbCrToPixels": {
                "params": {
                    "layout": "BGR",
                },
                "io_map": [
                    ["Y1_uint8", 0, 0],
                    ["Cb1_uint8", 0, 1],
                    ["Cr1_uint8", 0, 2],
                ],
            }
        },
        {"ConvertBGRToImage": {"image_format": "png"}},
    ],
    "tool_command_args": [
        {
            "name": "image",
            "data_type": "uint8",
            "shape": ["num_bytes"],
        }
    ],
    "target_opset": 16,
}
```

## Insert Beam Search Op

`InsertBeamSearch` chains two model components (for example, encoder and decoder) together by inserting beam search op in between them.

### Example Configuration
```json
{
    "type": "InsertBeamSearch",
    "no_repeat_ngram_size": 4
}
```

## ORT Performance Tuning
ONNX Runtime provides high performance across a range of hardware options through its Execution Providers interface for different execution
environments.
For each model running with each execution provider, there are settings that can be tuned (e.g. thread number, execution mode, etc) to
improve performance.
`OrtSessionParamsTuning` covers basic knobs that can be leveraged to find the best performance for your model and hardware.

### Example Configuration
```json
{
    "type": "OrtSessionParamsTuning",
    "data_config": "session_params_tuning_data_config",
    "batch_size": 1,
    "providers_list" : [
        [
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2147483648, // 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": true,
            },
        ],
        "CPUExecutionProvider",
    ],
    "enable_profiling": false
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py)
for an example implementation of `"user_script.py"` and `"calib_data_config/dataloader_config/type"`.

[1]: <https://onnxruntime.ai/docs/performance/quantization.html> "ONNX Runtime Quantization"
[2]: <https://onnxruntime.ai/docs/performance/quantization.html#dynamic-quantization> "Dynamic Quantization"
[3]: <https://onnxruntime.ai/docs/performance/quantization.html#static-quantization> "Static Quantization"

## Extract Adapters

LoRA, QLoRA and related techniques allow us to fine-tune a pre-trained model by adding a small number of trainable matrices called adapters. The same base model can be used for multiple tasks by adding different adapters for each task. To support using multiple adapters with the same optimized onnx model, the `ExtractAdapters` pass extracts the adapters weights from the model and saves them to a separate file. The model graph is then modified in one of the following ways:
- Adapters weights are set as external tensors pointing to a non-existent file. The onnx model is thus invalid by itself as it cannot be loaded. In order to create an inference session using this model, the adapter weights must be added to a sessions options object using `add_initializer` or `add_external_initializers`.
- Adapter weights are converted into model inputs. The onnx model is valid. During inference, the adapter weights must be provided as part of the inputs. We call them constant inputs here since these weights don't change between runs when using the one set of adapters.

### Example Configuration

a. As external initializers
```json
{
    "type": "ExtractAdapters",
    "make_inputs": false
}
```

b. As constant inputs with packed weights
```json
{
    "type": "ExtractAdapters",
    "make_inputs": true,
    "pack_inputs": true
}
```

Please refer to [ExtractAdapters](../../../reference/pass.rst#extract_adapters) for more details about the pass and its config parameters.

Olive also provides a command line tool to convert adapters saved after peft fine-tuning to a format compatible with a model that has been optimized with the `ExtractAdapters` pass. More details on the ``olive convert-adapters`` command can be found at [Command Line Tools](../../../reference/cli.rst).
