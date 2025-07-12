### How-tos
Find more details on specific Olive capabilities, such as quantization, running workflows on remote compute, model packaging, conversions, and more!

## Installation and Setup
- [How to install Olive](installation.html)


## Olive Command Lines

The Olive CLI provides a set of primitives such as `quantize`, `finetune`, `onnx-graph-capture`, `auto-opt` that enable you to *easily* optimize select models and experiment with different cutting-edge optimization strategies without the need to define workflows.

- [How to use the `olive auto-opt` command to take a PyTorch/Hugging Face model and turn it into an optimized ONNX model](cli/cli-auto-opt.html)
- [how to use the `olive finetune` command to create (Q)LoRA adapters](cli/cli-finetune.html)
- [How to use the `olive quantize` command to quantize your model with different precisions and techniques such as AWQ](cli/cli-quantize.html)
- [How to use the `olive run` command to execute an Olive workflow.](cli/cli-run.html>)

## Olive Python API

- [How to use the Python API to run Olive workflows programmatically](python_api.html)

## Customize Workflow (aka Recipes)

- [How to write a new workflow from scratch](configure-workflows/build-workflow.html)
- [How to define input model for a new workflow](configure-workflows/how-to-configure-model.html)
- [How to customize a pass parameters](configure-workflows/pass-configuration.html)
- [How to setup custom dataset for calibration and evaluation](configure-workflows/how-to-configure-data.html)
- [How to define evaluation metrics such as accuracy, latency, throughput, and your own custom metrics](configure-workflows/metrics-configuration.html)
- [How to package output model for deployment](configure-workflows/model-packaging.html)
- [How to define `host` or `target` systems](configure-workflows/systems.html>)
- [How to configure Olive `engine`](configure-workflows/engine-configuration.html)
