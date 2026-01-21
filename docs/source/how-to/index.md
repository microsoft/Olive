# How Tos

# Installation and Setup
- [How to install Olive](installation)

# Olive Command Lines

The Olive CLI provides a set of primitives such as `quantize`, `finetune`, `onnx-graph-capture`, `auto-opt` that enable you to *easily* optimize select models and experiment with different cutting-edge optimization strategies without the need to define workflows.

- [How to use the `olive optimize` command to optimize a Pytorch model](cli/cli-optimize)
- [How to use the `olive auto-opt` command to take a PyTorch/Hugging Face model and turn it into an optimized ONNX model](cli/cli-auto-opt)
- [how to use the `olive finetune` command to create (Q)LoRA adapters](cli/cli-finetune)
- [How to use the `olive quantize` command to quantize your model with different precisions and techniques such as AWQ](cli/cli-quantize)
- [How to use the `olive run` command to execute an Olive workflow.](cli/cli-run)

# Olive Python API

- [How to use Python API](python-api)

# Customize Workflow (aka Recipes)

- [How to write a new workflow from scratch](configure-workflows/build-workflow)
- [How to define input model for a new workflow](configure-workflows/how-to-configure-model)
- [How to customize a pass parameters](configure-workflows/pass-configuration)
- [How to setup custom dataset for calibration and evaluation](configure-workflows/how-to-configure-data)
- [How to define evaluation metrics such as accuracy, latency, throughput, and your own custom metrics](configure-workflows/metrics-configuration)
- [How to package output model for deployment](configure-workflows/model-packaging)
- [How to define `host` or `target` systems](configure-workflows/systems)
- [How to configure Olive `engine`](configure-workflows/engine-configuration)

# Extend Olive

- [Olive design overview](extending/design)
- [How to add a new Pass](extending/how-to-add-optimization-pass)
- [How to add a new task for ONNX export](extending/how-to-add-new-task.md)
- [How to add custom model evaluator](extending/custom-model-evaluator)
- [How to add custom scripts to load datasets](extending/custom-scripts)

<!-- Required by sphinx -->
```{toctree}
:maxdepth: 2
:hidden:

installation
cli/cli-optimize
cli/cli-auto-opt
cli/cli-finetune
cli/cli-quantize
cli/cli-run
python-api
configure-workflows/build-workflow
configure-workflows/how-to-configure-model
configure-workflows/pass-configuration
configure-workflows/how-to-configure-data
configure-workflows/metrics-configuration
configure-workflows/model-packaging
configure-workflows/systems
configure-workflows/engine-configuration
extending/design
extending/how-to-add-optimization-pass
extending/how-to-add-new-task
extending/custom-model-evaluator
extending/custom-scripts
```
