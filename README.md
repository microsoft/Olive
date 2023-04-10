# Olive

Olive is an easy-to-use hardware-aware model optimization tool that composes industry-leading techniques
across model compression, optimization, and compilation. Given a model and targeted hardware, Olive composes the best
suitable optimization techniques to output the most efficient model(s) for inferencing on cloud or edge, while taking
a set of constraints such as accuracy and latency into consideration.

Since every ML accelerator vendor implements their own acceleration tool chains to make the most of their hardware, hardware-aware
optimizations are fragmented. With Olive, we can:

Reduce engineering effort for optimizing models for cloud and edge: Developers are required to learn and utilize
multiple hardware vendor-specific toolchains in order to prepare and optimize their trained model for deployment.
Olive aims to simplify the experience by aggregating and automating optimization techniques for the desired hardware
targets.

Build up a unified optimization framework: Given that no single optimization technique serves all scenarios well,
Olive enables an extensible framework that allows industry to easily plugin their optimization innovations.  Olive can
efficiently compose and tune integrated techniques for offering a ready-to-use E2E optimization solution.

## Get Started and Resources
- Documentation: [https://microsoft.github.io/Olive](https://microsoft.github.io/Olive)
- Examples: [examples](./examples)

## Installation
We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Olive is installed using
pip.

Create a virtual/conda environment with the desired version of Python and activate it.

You will need to install a build of [**onnxruntime**](https://onnxruntime.ai). You can install the desired build separately but
public versions of onnxruntime can also be installed as extra dependencies during Olive installation.

### Install with pip
Olive is available for installation from PyPI.
```
pip install olive-ai
```
With onnxruntime (Default CPU):
```
pip install olive-ai[cpu]
```
With onnxruntime-gpu:
```
pip install olive-ai[gpu]
```
With onnxruntime-directml:
```
pip install olive-ai[directml]
```

### Optional Dependencies
Olive has optional dependencies that can be installed to enable additional features. These dependencies can be installed as extras:
- **azureml**: To enable AzureML integration. Packages: `azure-ai-ml, azure-identity`
- **docker**: To enable docker integration. Packages: `docker`
- **openvino**: To use OpenVINO related passes. Packages: `openvino==2022.3.0, openvino-dev[tensorflow,onnx]==2022.3.0`

## Contributing
We’d love to embrace your contribution to Olive. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

### Formatting
Olive uses pre-commit hooks to check and format code. To install the pre-commit hooks, run the following commands from the root of the repository:

```bash
# install pre-commit and other dev requirements
python -m pip install pre-commit
# install the git hook scripts
pre-commit install
# for the first time, run on all files
pre-commit run --all-files
```

Everytime, you make a git commit after the above, the hooks will automatically point out issues in code for changed files and fix them if possible.

## License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
