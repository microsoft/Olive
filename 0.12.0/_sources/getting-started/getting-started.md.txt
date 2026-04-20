---
hide:
- navigation
---

# Getting started

## Notebook available!

This quickstart is available as a Jupyter Notebook, which you can download and run on your own computer.

```{button-link} https://github.com/microsoft/Olive/blob/main/notebooks/olive_quickstart.ipynb
:color: primary
:outline:

Download Jupyter Notebook
```

## {fab}`python` Install with pip

We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


:::: {tab-set}

::: {tab-item} Windows/Linux/Mac CPU
If your machine only has a CPU device, install the following libraries:

```bash
pip install olive-ai[cpu]
pip install transformers
```
:::

::: {tab-item} Windows/Linux/Mac GPU
If your machine has a GPU device (e.g. CUDA), install the following libraries:

```bash
pip install olive-ai[gpu]
pip install transformers
```
:::

::::



```{seealso}
For more details on installing Olive from source and other installation options available, [read the installation guide](../how-to/installation.md).
```

## {octicon}`dependabot;1em` Automatic model optimization with Olive

Once you have installed Olive, next you'll run the `optimize` command that will automatically download and optimize Llama-3.2-1B-Instruct. After the model is downloaded, Olive will convert it into ONNX format, quantize (`int4`), and optimizing the graph. It takes around 60secs plus model download time (which will depend on your network bandwidth).
```bash
olive optimize \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --precision int4 \
    --output_path models/qwen
```

### More details on arguments

- The `model_name_or_path` can be either (a) the Hugging Face Repo ID for the model `{username}/{repo-name}` or (b) a path on local disk to the model or (c) an Azure AI Model registry ID.
- `output_path` is the path on local disk to store the optimized model.
- `precision` is the precision for the optimized model (`fp16`, `fp32`, `int4`, `int8`).

With the `optimize` command, you can change the input model to one that is available on Hugging Face or a model that resides on local disk. Olive, will go through the same process of *automatically* converting (to ONNX), optimizing the graph and quantizing the weights. The model can be optimized for different providers and devices using `provider` and `device` options respectively.

- `device` is the device the model will execute on - CPU/NPU/GPU.
- `provider` is the hardware provider of the device to inference the model on. For example, Nvidia CUDA (`CUDAExecutionProvider`), AMD (`MIGraphXExecutionProvider`, `ROCMExecutionProvider`), OpenVINO (`OpenVINOExecutionProvider`), Qualcomm (`QNNExecutionProvider`), Nvidia TensorRT (`TensorrtExecutionProvider`, `NvTensorRTRTXExecutionProvider`).
