# How To Write New Olive workflow

To convert, quantize, optimize and tune your model Olive needs information about your model. For example, how to load the model, name and shape of input tensors. You can also select the target hardware and list of optimizations you want to perform on the model. You can provide this information in a json file as an input to the Olive. In this document we will walk through how to generate such .json configuration from scratch.

We will focus on processing a Hugging Face model targeting CPU. After the .json configuration is prepared, one simple command will process the model and produce intended output model.

```bash
olive run --config my_model_processing_description.json
```

> **Note:**
> If you prefer to run Olive workflows programmatically or want to access optimization results from Python, refer to the [Python Interface documentation](../python_api.md) for details on using Olive from Python.


## Input Model

Let's use Phi-3.5 available on Hugging Face ( https://huggingface.co/microsoft/Phi-3.5-mini-instruct )

```json
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/Phi-3.5-mini-instruct"
    }
```
Olive supports a number of different input model types including **HuggingFace**, **Pytorch**, **ONNX**, **OpenVINO**, **QNN**, **TensorFlow**, and **Composite**. For detailed instructions on configuring your input model, see [how to configure input model](../configure-workflows/how-to-configure-model.md).

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

You can additionally select output directory, log severity level etc,. See [options](../../reference/options.md) for complete list of configuration option. Now you have the complete .json configuration that you can use.

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

## Run multiple builds

Use `builds` to run different pass pipelines or model components from one workflow configuration. Each named build
references passes from the top-level `passes` dictionary. The optional `_default` entry supplies shared build values.

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/Phi-3.5-mini-instruct"
    },
    "passes": {
        "convert": {
            "type": "OnnxConversion"
        },
        "optimize": {
            "type": "OrtTransformersOptimization"
        }
    },
    "max_concurrent_builds": 2,
    "builds": {
        "_default": {
            "output_dir": "models"
        },
        "convert-only": {
            "pipeline": ["convert"]
        },
        "optimized": {
            "pipeline": ["convert", "optimize"]
        }
    }
}
```

`_default.output_dir` is a parent directory, so the example writes to `models/convert-only` and
`models/optimized`. A named build can set its own `output_dir` to override that behavior.
Build names and output paths must use portable path components; Windows-reserved names and components ending in a
space or period are rejected during validation.

Builds run concurrently by default. Set the top-level `max_concurrent_builds` field to a positive integer to bound
parallelism, or set it to `1` to force serial execution. Use parallel execution only when the builds have sufficient
independent CPU, GPU, and memory resources. Passes are thread-safe by default; a pass that modifies process-global
state must set `thread_safe: false` in its package configuration. If any selected pass is not thread-safe, Olive runs
the entire multi-build workflow serially.

The optional `components` field selects model components before running a build's pipeline. Multi-build workflows
currently require a local host, and every build must have non-overlapping output and cache directories.

## Summary

Olive provides additional opportunity to configure system, data, evaluation metrics and more. See [How to customize configuration](#how-to-customize-configuration).
