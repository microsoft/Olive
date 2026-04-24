# How To Use `auto-opt` Command

The `olive auto-opt` command automatically optimizes a PyTorch/Hugging Face model into the ONNX format so that it runs with quality and efficiency on the ONNX Runtime.

## {octicon}`zap` Quickstart

The Olive automatic optimization command (`auto-opt`) can pull models from Hugging Face, Local disk, or the Azure AI Model Catalog. In this quickstart, you'll be optimizing [Llama-3.2-1B-Instruct from Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main). Llama 3.2 is a gated model and therefore you'll need to be signed into Hugging-Face to get access.

``` bash
huggingface-cli login --token {TOKEN}
```

```{tip}
Follow the [Hugging Face documentation for setting up User Access Tokens](https://huggingface.co/docs/hub/security-tokens)
```

Next, you'll run the `auto-opt` command that will automatically download and optimize Llama-3.2-1B-Instruct. After the model is downloaded, Olive will convert it into ONNX format, quantize (`int4`), and optimizing the graph. It takes around 60secs plus model download time (which will depend on your network bandwidth).

```bash
olive auto-opt \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --trust_remote_code \
    --output_path models/llama/ao \
    --device cpu \
    --provider CPUExecutionProvider \
    --use_ort_genai \
    --precision int4 \
    --log_level 1
```

### More details on arguments

- The `model_name_or_path` can be either (a) the Hugging Face Repo ID for the model `{username}/{repo-name}` or (b) a path on local disk to the model or (c) an Azure AI Model registry ID.
- `output_path` is the path on local disk to store the optimized model.
- `device` is the device the model will execute on - CPU/NPU/GPU.
- `provider` is the hardware provider of the device to inference the model on. For example, Nvidia CUDA (`CUDAExecutionProvider`), DirectML (`DmlExecutionProvider`), AMD (`MIGraphXExecutionProvider`, `ROCMExecutionProvider`), OpenVINO (`OpenVINOExecutionProvider`), Qualcomm (`QNNExecutionProvider`), TensorRT (`TensorrtExecutionProvider`).
- `precision` is the precision for the optimized model (`fp16`, `fp32`, `int4`, `int8`). Precisions lower than `fp32` will mean the model will be quantized using RTN method.
- `use_ort_genai` will create the configuration files so that you can consume the model using the Generate API for ONNX Runtime.

With the `auto-opt` command, you can change the input model to one that is available on Hugging Face - for example, to [HuggingFaceTB/SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct) - or a model that resides on local disk. Olive, will go through the same process of *automatically* converting (to ONNX), optimizing the graph and quantizing the weights. The model can be optimized for different providers and devices - for example, you can choose DirectML (for Windows) as the provider and target either the NPU, GPU, or CPU device.
