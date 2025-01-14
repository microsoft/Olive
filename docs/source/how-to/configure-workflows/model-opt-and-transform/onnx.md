# ONNX

[ONNX](https://onnx.ai/) is an open graph format to represent machine learning models. [ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.

Olive provides multiple transformations and optimizations based on various ONNX to improve model performance.

## Peeophole Optimizer

`OnnxPeepholeOptimizer` optimizes an ONNX model. The optimization process involves analyzing the structure of the ONNX model and identifying opportunities.

The `OnnxPeepholeOptimizer` leverages `onnxscript` (https://onnxscript.ai/tutorial/optimizer/optimize.html) and `onnxoptimizer`(https://github.com/onnx/optimizer) underneath.

| Optimization                      | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| **Constant Folding**              | Applies constant folding optimization to the model.                        |
| **Constant Propagation**          | Applies constant propagation optimization to the model. Applied as part of constant folding. |
| **Sequence Simplification**       | Simplifies Sequence-based ops (e.g., SequenceConstruct, ConcatFromSequence). Part of constant folding. |
| **Remove Unused Nodes**           | Removes unused nodes from the model.                                       |
| **Remove Unused Functions**       | Removes unused function protos from the model.                             |
| **Inline Functions with Unused Outputs** | Inlines function nodes with unused outputs.                                |
| **Inline Simple Functions**       | Inlines simple functions based on a node count threshold.                  |
| **Eliminate Nop Cast**            | Eliminates no-operation (nop) Casts.                                                |
| **Eliminate Nop Dropout**         | Eliminates no-operation Dropouts.                                                   |
| **Eliminate Nop Flatten**         | Eliminates no-operation Flattens.                                                   |
| **Extract Constant to Initializer** | Extracts constants to initializers.                                                 |
| **Eliminate If with Const Cond**  | Eliminates If nodes with constant conditions.                                       |
| **Eliminate Nop Monotone ArgMax** | Eliminates nop monotone ArgMax.                                                     |
| **Eliminate Nop Pad**             | Eliminates no-operation Pads.                                                       |
| **Eliminate Nop Concat**          | Eliminates no-operation Concats.                                                    |
| **Eliminate Nop Split**           | Eliminates no-operation Splits.                                                     |
| **Eliminate Nop Expand**          | Eliminates no-operation Expands.                                                    |
| **Eliminate Shape Gather**        | Eliminates Shape Gather operations.                                                 |
| **Eliminate Slice after Shape**   | Eliminates Slice nodes that occur after Shape nodes.                                |
| **Eliminate Nop Transpose**       | Eliminates no-operation Transposes.                                                |
| **Fuse Add Bias into Conv**       | Fuses Add operations as biases into Conv layers.                                    |
| **Fuse BN into Conv**             | Fuses BatchNormalization into Conv layers.                                          |
| **Fuse Consecutive Concats**      | Fuses consecutive Concat operations.                                                |
| **Fuse Consecutive LogSoftmax**   | Fuses consecutive LogSoftmax operations.                                            |
| **Fuse Consecutive Reduce+Unsqueeze** | Fuses consecutive Reduce and Unsqueeze operations.                                 |
| **Fuse Consecutive Squeezes**     | Fuses consecutive Squeeze operations.                                               |
| **Fuse Consecutive Transposes**   | Fuses consecutive Transpose operations.                                             |
| **Fuse MatMul+Add Bias into GEMM** | Fuses MatMul and Add operations into GEMM layers.                                   |
| **Fuse Pad into Conv**            | Fuses Pad operations into Conv layers.                                              |
| **Fuse Pad into Pool**            | Fuses Pad operations into Pool layers.                                              |
| **Fuse Transpose into GEMM**      | Fuses Transpose operations into GEMM layers.                                        |
| **Fuse Concat into Reshape**      | Fuses Concat operations into Reshape layers.                                        |
| **Eliminate Nop Reshape**         | Eliminates no-operation Reshapes.                                                   |
| **Eliminate Nop with Unit**       | Eliminates no-operation nodes with unit values.                                     |
| **Eliminate Common Subexpression** | Eliminates common sub-expressions.                                                 |
| **Fuse QKV**                      | Fuses query, key, and value layers in transformer models.                          |
| **Fuse Consecutive Unsqueezes**   | Fuses consecutive Unsqueeze operations.                                             |
| **Eliminate Deadend Nodes**       | Eliminates dead-end nodes.                                                          |
| **Eliminate Identity Nodes**      | Eliminates Identity nodes.                                                          |
| **Eliminate Shape Ops**           | Eliminates Shape operations where possible.                                         |
| **Fuse Consecutive Slices**       | Fuses consecutive Slice operations.                                                 |
| **Eliminate Unused Initializer**  | Eliminates unused initializers.                                                     |
| **Eliminate Duplicate Initializer** | Eliminates duplicate initializers.                                  |
| **Broadcast to MatMul**           | Converts broadcast patterns into MatMul operations for better efficiency.  |
| **Cast Constant of Shape**        | Simplifies constant casting for shape operations.                          |
| **GEMM to MatMul+Add**            | Converts GEMM operations into MatMul and Add for improved compatibility.   |
| **No-Op Removal**                 | Removes redundant or no-op operations in the computation graph.            |

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
