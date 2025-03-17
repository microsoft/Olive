# Auto Optimization

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

## Inference model using ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight package (available in many programming languages) that runs cross-platform. ORT enables you to infuse your AI models into your applications so that inference is handled *on-device*. The following code creates a simple console-based chat interface that inferences your optimized model.

:::: {tab-set}

::: {tab-item} Python
Copy-and-paste the code below into a new Python file called `app.py`:

```python
import onnxruntime_genai as og
import numpy as np
import os

model_folder = "models/llama/ao/model"

# Load the base model and tokenizer
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 200

# Encode the system prompt
system_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|>"
system_tokens = tokenizer.encode(system_prompt)

chat_template = "<|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# Keep asking for input prompts in a loop
while True:
    text = input("Prompt (Use quit() to exit): ")
    if not text:
        print("Error, input cannot be empty")
        continue

    if text == "quit()":
        break

    # Generate prompt (prompt template + input)
    prompt = f'{chat_template.format(input=text)}'

    # encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)

    # Create params and generator
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    # Append system and input tokens to the generator
    generator.append_tokens(system_tokens + input_tokens)

    print("")
    print("Output: ", end='', flush=True)
    # Stream the output
    try:
        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end='', flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")
    print()
    print()

    del generator
```
:::

::::

Run the code with:

```bash
python app.py
```
