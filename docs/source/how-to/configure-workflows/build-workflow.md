# Introduction

To convert, quantize, optimize and tune your model Olive needs information about your model. For example, how to load the model, name and shape of input tensors. You can also select the target hardware and list of optimizations you want to perform on the model. You can provide this information in a json file as an input to the Olive. In this document we will walk through how to generate such .json configuration from scratch.

We will focus on processing a Hugging Face model targeting CPU. After the .json configuration is prepared, one simple command will process the model and produce intended output model.

```bash
olive run --config my_model_processing_description.json
```

## Input Model

Let's use Phi-3.5 available on Hugging Face ( https://huggingface.co/microsoft/Phi-3.5-mini-instruct )

```json
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/Phi-3.5-mini-instruct"
    }
```
Olive supports a number of different input model types including **HuggingFace**, **Pytorch**, **ONNX**, **OpenVINO**, **QNN**, **SNPE**, **TensorFlow**, and **Composite**. For detailed instructions on configuring your input model, see [how to configure input model](../configure-workflows/how-to-configure-model.md).

## Passes to apply

Olive can apply various transformations and optimizations, also known as passes, on the input model. Let's apply ONNX conversion and Graph Surgery passes to convert the model to ONNX and apply select graph transformations.

```json
    "passes" : {
        "mb": {
            "type": "ModelBuilder",
            "precision": "int4",
            "int4_block_size": 32,
            "int4_accuracy_level": 4,
            "int4_op_types_to_quantize": [ "MatMul", "Gather" ],
            "save_as_external_data": true
        },
        "gs": {
            "type": "GraphSurgeries",
            "surgeries": [
                { "surgeon": "RemoveRopeMultiCache" },
                { "surgeon": "AttentionMaskToSequenceLengths" },
                { "surgeon": "SimplifiedLayerNormToL2Norm" }
            ],
            "save_as_external_data": true
        }
    }
```
Here we are using ModelBuilder to convert the model to ONNX and applying three different graph surgeries on the converted ONNX model. For detailed instructions on configuring passes, see [how to configure pass](../configure-workflows/pass-configuration.md). For a complete list of available passes, see [passes](../../reference/pass.rst).

## Complete .json configuration

You can additionally select output directory, log severity level etc,. See [options](../../reference/options.html) for complete list of configuration option. Now you have the complete .json configuration that you can use.

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/Phi-3.5-mini-instruct",
        "load_kwargs": { "trust_remote_code": true }
    },
   "passes" : {
        "mb": {
            "type": "ModelBuilder",
            "precision": "int4",
            "int4_block_size": 32,
            "int4_accuracy_level": 4,
            "int4_op_types_to_quantize": [ "MatMul", "Gather" ],
            "save_as_external_data": true
        },
        "gs": {
            "type": "GraphSurgeries",
            "surgeries": [
                { "surgeon": "RemoveRopeMultiCache" },
                { "surgeon": "AttentionMaskToSequenceLengths" },
                { "surgeon": "SimplifiedLayerNormToL2Norm" }
            ],
            "save_as_external_data": true
        }
    },
    "log_severity_level" : 1,
    "output_dir" : "models/phi3_5"
}
```

## Summary

Olive provides additional opportunity to configure system, data, evaluation metrics and more. See [How to customize configuration](#how-to-customize-configuration)
