# Installation

We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Olive is installed using
pip.

Create a virtual/conda environment with the desired version of Python and activate it.

You will need to install a build of [**onnxruntime**](https://onnxruntime.ai). You can install the desired build separately but
public versions of onnxruntime can also be installed as extra dependencies during Olive installation.

## Install with pip

Olive is available for installation from PyPI.

```bash
pip install olive-ai
```

## Optional Dependencies
Olive has optional dependencies that can be installed to enable additional features. Please refer to
[Olive package config](https://github.com/microsoft/Olive/blob/main/olive/olive_config.json) for the list of extras
and their dependencies.


## Install from source

Install the latest `main` version of Olive from source. Please note that this is a development version and may not be stable.

```bash
pip install git+https://github.com/microsoft/Olive
```

With onnxruntime (Default CPU):
```bash
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[cpu]
```
With onnxruntime-gpu:

```bash
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[gpu]
```
With onnxruntime-directml:

```bash
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[directml]
```

## Editable install

If you want contribute to Olive and test your code, you can install Olive in editable mode.

Clone the repository and install Olive with the following commands:

```bash
git clone https://github.com/microsoft/Olive
cd Olive
pip install -e .
```
