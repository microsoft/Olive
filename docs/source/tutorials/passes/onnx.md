# ONNX related – General

Olive provides multiple Passes that execute optimization tools related to ONNX. [ONNX](https://onnx.ai/) is
an open format built to represent machine learning models. [ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform machine-learning
model accelerator, with a flexible interface to integrate hardware-specific libraries.

Olive provides easy access to the model optimization tools available in ONNX Runtime.

## Model Conversion
The user might not have a model ready in the ONNX format. `OnnxConversion` converts PyTorch models to ONNX using
[torch.onnx](https://pytorch.org/docs/stable/onnx.html).

Please refer to [OnnxConversion](onnx_conversion) for more details about the pass and its config parameters.

### Example Configuration
```json
 {
    "type": "OnnxConversion",
    "config": {
        "target_opset": 13
    }
 }
```

## Model Optimizer
`OnnxModelOptimizer` optimizes an ONNX model by fusing nodes. Fusing nodes involves merging multiple nodes in a model into a single node to
reduce the computational cost and improve the performance of the model.
The optimization process involves analyzing the structure of the ONNX model and identifying nodes that can be fused.

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
    "config": {"model_type": "bert"}
}
```
## Pre/Post Processing Ops
'PrePostProcessing' inserts pre and post processing ops into the ONNX graph.

### Example Configuration
```json
{
    "type": "PrePostProcessing",
    "config": {"tool_command": "superresolution"}
}
```
## Post Training Quantization (PTQ)
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

### Example Configuration
a. Tune the parameters of the OlivePass with pre-defined searchable values
```json
{
    "type": "OnnxQuantization",
    "config": {
        "user_script": "./user_script.py",
        "dataloader_func": "glue_calibration_reader"
    }
}
```

b. Select parameters to tune
```json
{
    "type": "OnnxQuantization",
    "config": {
        // select per_channel to tune with "SEARCHABLE_VALUES".
        // other parameters will use the default value, not to be tuned.
        "per_channel": "SEARCHABLE_VALUES",
        "user_script": "./user_script.py",
        "dataloader_func": "glue_calibration_reader",
    },
    "disable_search": true
}
```

c. Use default values of the OlivePass (no tuning in this way)
```json
{
    "type": "OnnxQuantization",
    "config": {
        // set per_channel to "DEFAULT_VALUE"
        "per_channel": "DEFAULT_VALUE",
        "user_script": "./user_script.py",
        "dataloader_func": "glue_calibration_reader",
    },
    "disable_search": true
}
```

d. Specify parameters with user defined values
```json
"onnx_quantization": {
    "type": "OnnxQuantization",
    "config": {
        // set per_channel to True.
        "per_channel": true,
        "user_script": "./user_script.py",
        "dataloader_func": "glue_calibration_reader",
    },
    "disable_search": true
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert_ptq_cpu/user_script.py)
for an example implementation of `"user_script.py"` and `"glue_calibration_reader"`.

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
    "config": {
        "user_script": "user_script.py",
        "dataloader_func": "create_dataloader",
        "batch_size": 1
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert_ptq_cpu/user_script.py)
for an example implementation of `"user_script.py"` and `"create_dataloader"`.

[1]: <https://onnxruntime.ai/docs/performance/quantization.html> "ONNX Runtime Quantization"
[2]: <https://onnxruntime.ai/docs/performance/quantization.html#dynamic-quantization> "Dynamic Quantization"
[3]: <https://onnxruntime.ai/docs/performance/quantization.html#static-quantization> "Static Quantization"

## Float16 Conversion

Converting a model to use Float16 instead of Float32 can decrease the model size and improve performance on some GPUs. The `OnnxFloatToFloat16` pass wraps [onnxconverter_common.float16.convert_float_to_float16](https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py#L111), which convert most nodes/operators to use Float16 instead of Float32.

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
    "config": {
        // Don't convert input/output nodes to Float16
        "keep_io_types": true
    }
}
```

See [Float16 Conversion](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html#float16-conversion) for more detailed description of the available configuration parameters.

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
    "config": {
        "op_block_list": [
            "Add",
            "LayerNormalization",
            "SkipLayerNormalization",
            "FastGelu",
            "EmbedLayerNormalization",
        ]
    }
}
```
