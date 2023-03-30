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
a. Provide input shapes
```json
 {
    "type": "OnnxConversion",
    "config": {
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "input_shapes": [[1, 128], [1, 128], [1, 128]],
        "input_types": ["int64", "int64", "int64"],
        "output_names": ["output"],
        "target_opset": 13
    }
 }
```

b. Provide custom input tensor function
```json
{
    "type": "OnnxConversion",
    "config": {
        "user_script": "user_script.py",
        "input_tensor_func": "create_input_tensors",
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "output_names": ["output"],
        "target_opset": 13
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert_ptq_cpu/user_script.py)
for an example implementation of `"user_script.py"` and `"create_input_tensors"`.

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
a. Tune the parameters of the OlivePass with pre-defined search space
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
        // set per_channel to "DEFAULT" value.
        "per_channel": "DEFAULT",
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
