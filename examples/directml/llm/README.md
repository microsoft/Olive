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

If you only want to run the inference sample (possible after the model has been optimized), run the `run_llm_io_binding.py` helper script:

```
python run_llm_io_binding.py --model_type=llama-2-7b-chat --prompt=<any_prompt_you_choose>
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
