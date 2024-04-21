# LLM Optimization with DirectML <!-- omit in toc -->

This sample shows how to optimize many LLMs to run with ONNX Runtime and DirectML.

**Contents**:
- [Setup](#setup)
- [Conversion to ONNX and Latency Optimization](#conversion-to-onnx-and-latency-optimization)

# Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

1. Install Olive

```
pip install -e .
```

2. Install the requirements

```
cd Olive/examples/directml/llm
pip install -r requirements.txt
```

3. (Only for LLaMA 2) Request access to the LLaMA 2 weights at the HuggingFace's [llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) or [llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) repositories.


# Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `llm.py` helper script:

1. To convert LLaMA 2:

```
python llm.py --model_type=llama-2-7b-chat
```

2. To convert Mistral:

```
python llm.py --model_type=mistral-7b-chat
```

To see the full list of models available to convert, just run the following command:

```
python llm.py --help
```

The first time this script is invoked can take some time since it will need to download the model weights from HuggingFace. The LLMs can be very large and the optimization process is resource intensive (it can sometimes take more than 200GB of ram). You can still optimize the model on a machine with less memory, but you'd have to increase your paging file size accordingly. **If the optimization process aborts without an error message or tells you that python stopped working, it is likely due to an OOM error and you should increase your paging file size.**

Once the script successfully completes, the optimized ONNX pipeline will be stored under `models/optimized/<model_name>`.

Note: When converting mistral, you will see the following error: `failed in shape inference <class 'AssertionError'>`. This is caused by the MultiHeadAttention operator not support Multi Query Attention, but in our case it doesn't matter since it will be converted to GroupQueryAttention at the end of the optimization process. You can safely ignore this error.

If you only want to run the inference sample (possible after the model has been optimized), run the `run_llm_io_binding.py` helper script:

```
python run_llm_io_binding.py --model_type=llama-2-7b-chat --prompt=<any_prompt_you_choose>
```

# AWQ Quantization

AWQ quantization is very resource intensive and currently requires a decent Nvidia GPU (at least 4090). Even with an RTX 4090, it will take several hours. If you want to quantize a model, install a fork of the Intel Neural Compressor and convert the model with the `--quant_strategy=awq` option:

```
pip install git+https://github.com/PatriceVignola/neural-compressor@6a05c1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python llm.py --model_type=mistral-7b-chat --quant_strategy=awq
```

# Running the Interactive Chat App Example

Before running the chat app, you need to need to install the gradio requirements:

```
pip install gradio==3.42.0
```

Then, simply go run the chat app from the current directory and copy-paste the link in your browser:

```
python .\chat_app\app.py
```

Note that the chat app is for demo purposes only and only has basic functionalities. It only supports argmax sampling and will simply stop working when the cache is full.

# Supporting a New Model

To support a new model with similar architecture to the existing LLaMA/Mistral/Phi-2 models, simply add it to the lists in `model_type_mapping.py`, as well as adding the chat template in `chat_templates.py`. The models in this list have all been tested with DirectML and we confirmed that they are accurate and fast, but it doesn't mean that other models won't work. In fact, most models that we added to the list except for the initial LLaMA, Mistral and Phi-2 models worked without changing anything in the architecture.

# Coming Soon

You may have noticed that we do not use the modeling files from the transformers library directly when converting models. This is because we found out that some of their pytorch models do not have a pattern that is able to get fused by the onnxruntime transformers optimizer, but this is only a temporary solution. The long-term plan is to get rid of this custom `decoder_model.py` architecture and instead consume models directly from the transformers libraries. In fact, many models already work with it out of the box, but some of them need changes in onnxruntime to make sure that the fusions are applied.
