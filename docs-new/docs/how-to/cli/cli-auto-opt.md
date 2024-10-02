# Auto Optimization

The `olive auto-opt` command automatically optimizes a PyTorch/Hugging Face model into the ONNX format so that it runs with quality and efficiency on the ONNX Runtime.

## :material-clock-fast: Quickstart

The Olive automatic optimization command (`auto-opt`) can pull models from Hugging Face, Local disk, or the Azure AI Model Catalog. In this getting started guide, you'll be optimizing [Llama-3.2-1B-Instruct from Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main). Llama 3.2 is a gated model and therefore you'll need to be signed into Hugging-Face to get access. 

``` bash
huggingface-cli login --token {TOKEN} # (1)!
```

1. Follow the [Hugging Face documentation for setting up User Access Tokens](https://huggingface.co/docs/hub/security-tokens)

The `olive auto-opt` command that will automatically download and optimize Llama-3.2-1B-Instruct. After the model is downloaded, Olive will convert it into ONNX format, quantize (`int4`), and optimizing the graph. It takes around 60secs plus model download time (which will depend on your network bandwidth).

``` bash
olive auto-opt \ 
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \  # (1)!
    --trust_remote_code \ 
    --output_path optimized-model \ # (2)!
    --device cpu \ # (3)!
    --providers CPUExecutionProvider \ # (4)!
    --precisions int4 \ # (5)!
    --log_level 1 # (6)!
```

1. Can be either (a) the Hugging Face Repo ID for the model` {username}/{repo-name}` or (b) a path on local disk to the model or (c) an Azure AI Model registry ID.
2. The output path on local disk to store the optimized model.
3. The device type to model will execute on - CPU/NPU/GPU.
4. The hardware provider - for example Nvidia CUDA (`CUDAExecutionProvider`), DirectML (`DmlExecutionProvider`), AMD (`MIGraphXExecutionProvider`, `ROCMExecutionProvider`), OpenVINO (`OpenVINOExecutionProvider`), Qualcomm (`QNNExecutionProvider`), TensorRT (`TensorrtExecutionProvider`).
5. The precision of the optimized model (`fp16`, `fp32`, `int4`, `int8`).
6. The logging level. 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL.

With the `auto-opt` command, you can change the input model to one that is available on Hugging Face - for example, to [HuggingFaceTB/SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct) - or a model that resides on local disk. Olive, will go through the same process of *automatically* converting (to ONNX), optimizing the graph and quantizing the weights. The model can be optimized for different providers and devices - for example, you can choose DirectML (for Windows) as the provider and target either the NPU, GPU, or CPU device.