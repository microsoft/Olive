# ONNX

[ONNX](https://onnx.ai/) is an open graph format to represent machine learning models. [ONNX Runtime](https://onnxruntime.ai/docs/) is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.

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

## Inputs/Outputs DataType Conversion

In certain environments, such as Onnxruntime WebGPU, Float32 logits are preferred. The `OnnxIODataTypeConverter` pass enables conversion of model inputs and outputs to a specified data type. This is particularly useful for converting between data types such as Float16 and Float32, or any other supported ONNX data types.

### Example Configuration

The simplest configuration converts all inputs and outputs from Float16 (source_dtype = 10) to Float32 (target_dtype = 1), which is suitable for many models:

```json
{
    "type": "OnnxIODataTypeConverter",
    "source_dtype": 10,
    "target_dtype": 1
}
```

### Datatype Mapping

The `source_dtype` and `target_dtype` are integers corresponding to ONNX data types. You can find the complete mapping in the ONNX protobuf definition [here](https://github.com/onnx/onnx/blob/96a0ca4374d2198944ff882bd273e64222b59cb9/onnx/onnx.proto3#L503-L551).

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
