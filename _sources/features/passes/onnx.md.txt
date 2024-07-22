# ONNX

[ONNX](https://onnx.ai/) is an open graph format to represent machine learning models. [ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.

Olive provides multiple transformations and optimizations based on various ONNX to improve model performance.

## Model Conversion
The `OnnxConversion` pass converts PyTorch models to ONNX using
[torch.onnx](https://pytorch.org/docs/stable/onnx.html).

Please refer to [OnnxConversion](onnx_conversion) for more details about the pass and its config parameters.

Besides, if you want to convert an existing ONNX model with another target opset, you can use [OnnxOpVersionConversion](onnx_op_version_conversion) pass, similar configs with above case:

### Example Configuration
```json
 {
    "type": "OnnxConversion",
    "target_opset": 13
 },
 {
    "type": "OnnxOpVersionConversion",
    "target_opset": 14
 }
```

For generative models, the alternative conversion pass [ModelBuilder](model_builder) that integrates the
[ONNX Runtime Generative AI](https://github.com/microsoft/onnxruntime-genai) module can be used.

Please refer to [ModelBuilder](model_builder) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "ModelBuilder",
    "precision": "int4"
}
```

## Model Optimizer
`OnnxModelOptimizer` optimizes an ONNX model by fusing nodes. Fusing nodes involves merging multiple nodes in a model into a single node to
reduce the computational cost and improve the performance of the model. The optimization process involves analyzing the structure of the ONNX model and identifying nodes that can be fused.

Also, inserts a `Cast` operation for cases where `ArgMax` input isn't supported on the device. For example, on GPU inferencing, TensorProto.INT64 isn't supported so a `Cast` operator inserted to cast the inputs to TensorProto.INT32.

Please refer to [OnnxModelOptimizer](onnx_model_optimizer) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OnnxModelOptimizer"
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

Please refer to [OrtTransformersOptimization](ort_transformers_optimization) for more details about the pass and its config parameters.

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

## Quantize with onnxruntime
[Quantization][1] is a technique to compress deep learning models by reducing the precision of the model weights from 32 bits to 8 bits. This
technique is used to reduce the memory footprint and improve the inference performance of the model. Quantization can be applied to the
weights of the model, the activations of the model, or both.

There are two ways to quantize a model in onnxruntime:
1. [Dynamic Quantization][2]:
    Dynamic quantization calculates the quantization parameters (scale and zero point) for activations dynamically, which means there is no
    any requirement for the calibration dataset.

    These calculations increase the cost of inference, while usually achieve higher accuracy comparing to static ones.


2. [Static Quantization][3]:
    Static quantization method runs the model using a set of inputs called calibration data. In this way, user must provide a calibration
    dataset to calculate the quantization parameters (scale and zero point) for activations before quantizing the model.

Olive consolidates the dynamic and static quantization into a single pass called `OnnxQuantization`, and provide the user with the ability to
tune both quantization methods and hyperparameter at the same time.
If the user desires to only tune either of dynamic or static quantization, Olive also supports them through `OnnxDynamicQuantization` and
`OnnxStaticQuantization` respectively.

Please refer to [OnnxQuantization](onnx_quantization), [OnnxDynamicQuantization](onnx_dynamic_quantization) and
[OnnxStaticQuantization](onnx_static_quantization) for more details about the passes and their config parameters.

**Note:** If target execution provider is QNN EP, the model might need to be preprocessed before quantization. Please refer to [QnnPreprocess](qnn_preprocess) for more details about the pass and its config parameters.
This preprocessing step fuses operators unsupported by QNN EP and inserts necessary operators to make the model compatible with QNN EP.

### Example Configuration
a. Tune the parameters of the OlivePass with pre-defined searchable values
```json
{
    "type": "OnnxQuantization",
    "user_script": "./user_script.py",
    "dataloader_func": "glue_calibration_reader"
}
```

b. Select parameters to tune
```json
{
    "type": "OnnxQuantization",
    // select per_channel to tune with "SEARCHABLE_VALUES".
    // other parameters will use the default value, not to be tuned.
    "per_channel": "SEARCHABLE_VALUES",
    "user_script": "./user_script.py",
    "dataloader_func": "glue_calibration_reader",
    "disable_search": true
}
```

c. Use default values of the OlivePass (no tuning in this way)
```json
{
    "type": "OnnxQuantization",
    // set per_channel to "DEFAULT_VALUE"
    "per_channel": "DEFAULT_VALUE",
    "user_script": "./user_script.py",
    "dataloader_func": "glue_calibration_reader"
}
```

d. Specify parameters with user defined values
```json
"onnx_quantization": {
    "type": "OnnxQuantization",
    // set per_channel to True.
    "per_channel": true,
    "user_script": "./user_script.py",
    "dataloader_func": "glue_calibration_reader",
    "disable_search": true
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py)
for an example implementation of `"user_script.py"` and `"glue_calibration_reader"`.

check out [this file](https://github.com/microsoft/Olive/tree/main/examples/bert#bert-optimization-with-intel-neural-compressor-ptq-on-cpu) for an example for Intel® Neural Compressor quantization.

## Quantize with Intel® Neural Compressor
In addition to the default onnxruntime quantization tool, Olive also integrates [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Intel® Neural Compressor is a model compression tool across popular deep learning frameworks including TensorFlow, PyTorch, ONNX Runtime (ORT) and MXNet, which supports a variety of powerful model compression techniques, e.g., quantization, pruning, distillation, etc. As a user-experience-driven and hardware friendly tool, Intel® Neural Compressor focuses on providing users with an easy-to-use interface and strives to reach “quantize once, run everywhere” goal.

Olive consolidates the Intel® Neural Compressor dynamic and static quantization into a single pass called `IncQuantization`, and provide the user with the ability to
tune both quantization methods and hyperparameter at the same time.
If the user desires to only tune either of dynamic or static quantization, Olive also supports them through `IncDynamicQuantization` and
`IncStaticQuantization` respectively.

### Example Configuration
```json
"inc_quantization": {
    "type": "IncStaticQuantization",
    "user_script": "user_script.py",
    "approach": "weight_only",
    "weight_only_config": {
        "bits": 4,
        "algorithm": "GPTQ"
    },
    "dataloader_func": "calib_dataloader",
    "calibration_sampling_size": [8],
    "save_as_external_data": true,
    "all_tensors_to_one_file": true
}
```

Please refer to [IncQuantization](inc_quantization), [IncDynamicQuantization](inc_dynamic_quantization) and
[IncStaticQuantization](inc_static_quantization) for more details about the passes and their config parameters.

## Quantize with AMD Vitis AI Quantizer
Olive also integrates [AMD Vitis AI Quantizer](https://github.com/microsoft/Olive/blob/main/olive/passes/onnx/vitis_ai/quantize.py) for quantization.

The Vitis™ AI development environment accelerates AI inference on AMD® hardware platforms. The Vitis AI quantizer can reduce the computing complexity by converting the 32-bit floating-point weights and activations to fixed-point like INT8. The fixed-point network model requires less memory bandwidth, thus providing faster speed and higher power efficiency than the floating-point model.
Olive consolidates the Vitis™ AI quantization into a single pass called VitisAIQuantization which supports power-of-2 scale quantization methods and supports Vitis AI Execution Provider.

### Example Configuration
```json
"vitis_ai_quantization": {
    "type": "VitisAIQuantization",
    "calibrate_method":"NonOverflow",
    "quant_format":"QDQ",
    "activation_type":"QUInt8",
    "weight_type":"QInt8",
    "user_script": "user_script.py",
    "data_dir": "data",
    "dataloader_func": "resnet_calibration_reader"
}
```
Please refer to [VitisAIQuantization](vitis_ai_quantization) for more details about the pass and its config parameters.

## ORT Performance Tuning
ONNX Runtime provides high performance across a range of hardware options through its Execution Providers interface for different execution
environments.
For each model running with each execution provider, there are settings that can be tuned (e.g. thread number, execution mode, etc) to
improve performance.
`OrtPerfTuning` covers basic knobs that can be leveraged to find the best performance for your model and hardware.

### Example Configuration
```json
{
    "type": "OrtPerfTuning",
    "data_config": "perf_tuning_data_config",
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
for an example implementation of `"user_script.py"` and `"create_dataloader"`.

[1]: <https://onnxruntime.ai/docs/performance/quantization.html> "ONNX Runtime Quantization"
[2]: <https://onnxruntime.ai/docs/performance/quantization.html#dynamic-quantization> "Dynamic Quantization"
[3]: <https://onnxruntime.ai/docs/performance/quantization.html#static-quantization> "Static Quantization"

## Float16 Conversion

Converting a model to use Float16 instead of Float32 can decrease the model size and improve performance on some GPUs. The `OnnxFloatToFloat16` pass the [float16 converter from onnxruntime](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/float16.py) to convert the model to float16, which convert most nodes/operators to use Float16 instead of Float32.

Conversion to Float16 is often exposed at multiple stages of optimization, including model conversion and transformer optimization. This stand-alone pass is best suited for models that are not transformer architectures, where fusions may rely on a specific data types in node patterns.

### Example Configuration

a. The most basic configuration, which is suitable for many models, leaves all configuration options set to their default values:
```json
{
    "type": "OnnxFloatToFloat16"
}
```

b. More fine-grained control of the conversion conditions is also possible:
```json
{
    "type": "OnnxFloatToFloat16",
    // Don't convert input/output nodes to Float16
    "keep_io_types": true
}
```

See [Float16 Conversion](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html#float16-conversion) for more detailed description of the available configuration parameters.

## Inputs/Outputs Float16 to Float32 Conversion

Certain environments such as Onnxruntime WebGPU prefers Float32 logits. The `OnnxIOFloat16ToFloat32` pass converts the inputs and outputs to use Float32 instead of Float16.

### Example Configuration

a. The most basic configuration, which is suitable for many models, leaves all configuration options set to their default values:
```json
{
    "type": "OnnxIOFloat16ToFloat32"
}
```

## Mixed Precision Conversion
Converting model to mixed precision.

If float16 conversion is giving poor results, you can convert most of the ops to float16 but leave some in float32. The `OrtMixedPrecision` pass finds a minimal set of ops to skip while retaining a certain level of accuracy.

The default value for `op_block_list` is `["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "Relu", "Add"]`.

### Example Configuration

a. The most basic configuration, which is suitable for many models, leaves all configuration options set to their default values:
```json
{
    "type": "OrtMixedPrecision"
}
```

b. More fine-grained control of the conversion conditions is also possible:
```json
{
    "type": "OrtMixedPrecision",
    "op_block_list": [
        "Add",
        "LayerNormalization",
        "SkipLayerNormalization",
        "FastGelu",
        "EmbedLayerNormalization",
    ]
}
```

## Convert dynamic shape to fixed shape

In qnn, snpe and other mobile inference scenarios, the input shape of the model is often fixed. The `DynamicToFixedShape` pass converts the dynamic shape of the model to a fixed shape.

For example, often models have a dynamic batch size so that training is more efficient. In mobile scenarios the batch generally has a size of 1. Making the batch size dimension ‘fixed’ by setting it to 1 may allow NNAPI and CoreML to run of the model.

The helper can be used to update specific dimensions, or the entire input shape.

### Example Configuration

a. Making a symbolic dimension fixed
```json
{
    "type": "DynamicToFixedShape",
    "input_dim": ["batch_size"],
    "dim_value": [1]
}
```

b. Making the entire input shape fixed
```json
{
    "type": "DynamicToFixedShape",
    "input_name": ["input"],
    "input_shape": [[1, 3, 224, 224]]
}
```

Note: The `input_dim` and `dim_value` should have the same length, and the `input_name` and `input_shape` should have the same length. Also the `input_dim & dim_value` and `input_name & input_shape` should be exclusive to each other, user cannot specify both of them at the same time.

More details about the pass and its config parameters can be found [here](https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html).

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

Please refer to [ExtractAdapters](extract_adapters) for more details about the pass and its config parameters.

Olive also provides a command line tool to export adapters saved after peft fine-tuning to a format compatible with a model that has been optimized with the `ExtractAdapters` pass. More details on the ``olive export-adapters`` command can be found at [Command Line Tools](command_line_tools).
