# Auto Optimization

The `olive auto-opt` command automatically optimizes a PyTorch/Hugging Face model into the ONNX format so that it runs with quality and efficiency on the ONNX Runtime.

## {octicon}`zap` Quickstart

The Olive automatic optimization command (`auto-opt`) can pull models from Hugging Face, Local disk, or the Azure AI Model Catalog. In this getting started guide, you'll be optimizing [Llama-3.2-1B-Instruct from Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main). Llama 3.2 is a gated model and therefore you'll need to be signed into Hugging-Face to get access.

``` bash
huggingface-cli login --token {TOKEN} # (1)
```

**Annotations:**

1. Follow the [Hugging Face documentation for setting up User Access Tokens](https://huggingface.co/docs/hub/security-tokens)

The `olive auto-opt` command that will automatically download and optimize Llama-3.2-1B-Instruct. After the model is downloaded, Olive will convert it into ONNX format, quantize (`int4`), and optimizing the graph. It takes around 60secs plus model download time (which will depend on your network bandwidth).

``` bash
olive auto-opt \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \  # (1)
    --trust_remote_code \
    --output_path optimized-model \ # (2)
    --device cpu \ # (3)
    --providers CPUExecutionProvider \ # (4)
    --precisions int4 \ # (5)
    --log_level 1 # (6)
```

**Annotations:**

1. Can be either (a) the Hugging Face Repo ID for the model` {username}/{repo-name}` or (b) a path on local disk to the model or (c) an Azure AI Model registry ID.
2. The output path on local disk to store the optimized model.
3. The device type to model will execute on - CPU/NPU/GPU.
4. The hardware provider - for example Nvidia CUDA (`CUDAExecutionProvider`), DirectML (`DmlExecutionProvider`), AMD (`MIGraphXExecutionProvider`, `ROCMExecutionProvider`), OpenVINO (`OpenVINOExecutionProvider`), Qualcomm (`QNNExecutionProvider`), TensorRT (`TensorrtExecutionProvider`).
5. The precision of the optimized model (`fp16`, `fp32`, `int4`, `int8`).
6. The logging level. 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL.

With the `auto-opt` command, you can change the input model to one that is available on Hugging Face - for example, to [HuggingFaceTB/SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct) - or a model that resides on local disk. Olive, will go through the same process of *automatically* converting (to ONNX), optimizing the graph and quantizing the weights. The model can be optimized for different providers and devices - for example, you can choose DirectML (for Windows) as the provider and target either the NPU, GPU, or CPU device.

### <span class="onnx-icon"></span> Inference model using ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight package (available in many programming languages) that runs cross-platform. ORT enables you to infuse your AI models into your applications so that inference is handled *on-device*. The following code creates a simple console-based chat interface that inferences your optimized model.

:::: {tab-set}

::: {tab-item} Python

Copy-and-paste the code below into a new Python file called `app.py`:

```python
import onnxruntime_genai as og
import numpy as np
import os

model_folder = "optimized-model/model"

# Load the base model and tokenizer
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 200
search_options['past_present_share_buffer'] = False

chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

text = input("Input: ")

# Keep asking for input phrases
while text != "exit":
    if not text:
        print("Error, input cannot be empty")
        exit

    # generate prompt (prompt template + input)
    prompt = f'{chat_template.format(input=text)}'

    # encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    print("Output: ", end='', flush=True)
    # stream the output
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end='', flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    print()
    text = input("Input: ")
```
:::

::::

Run the code with:

```bash
python app.py
```
