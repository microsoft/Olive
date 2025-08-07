# Finetune

The `olive finetune` command will finetune a PyTorch/Hugging Face model and output a Hugging Face PEFT adapter. If you want to convert the PEFT adapter into a format for the ONNX Runtime, you can execute the `olive generate-adapter` command after finetuning.

## {octicon}`zap` Quickstart

The following example shows how to finetune [Llama-3.2-1B-Instruct from Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) either using your local computer (if you have a GPU device) or using remote compute via Azure AI integration with Olive.

:::: {tab-set}

::: {tab-item} Local

```{Note}
You'll need a GPU device on your local machine to fine-tune a model.
```

```bash
olive finetune \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --trust_remote_code \
    --output_path models/llama/ft \
    --data_name xxyyzzz/phrase_classification \
    --text_template "<|start_header_id|>user<|end_header_id|>\n{phrase}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{tone}" \
    --method lora \
    --max_steps 100 \
    --log_level 1
```
:::

::: {tab-item} Azure AI

You can fine-tune on remote Azure ML compute by updating the placeholders (`{}`) in the following code snippet with your workspace, resource group and compute name details. Read the [How to create a compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2&tabs=azure-studio) article for more details on setting up a GPU cluster in Azure ML.

```bash
olive finetune \
    --model_name_or_path azureml://registries/azureml-meta/models/Llama-3.2-1B/versions/2 \
    --trust_remote_code \
    --output_path models/llama/ft \
    --data_name xxyyzzz/phrase_classification \
    --text_template "<|start_header_id|>user<|end_header_id|>\n{phrase}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{tone}" \
    --method lora \
    --max_steps 100 \
    --log_level 1 \
    --resource_group {RESOURCE_GROUP_NAME} \
    --workspace_name {WORKSPACE_NAME} \
    --aml_compute {COMPUTE_NAME}
```

You can download the model artifact using the Azure ML CLI:

```bash
az ml job download --name {JOB_ID} --resource-group {RESOURCE_GROUP_NAME} --workspace-name {WORKSPACE_NAME} --all
```
:::

::::

### {octicon}`dependabot;1em` Auto-Optimize the model and adapters

If you would like your fine-tuned model to run on the ONNX Runtime, you'll can execute the `olive auto-opt` command to produce an optimized ONNX model and adapter, using

```bash
olive auto-opt \
   --model_name_or_path models/llama/ft/model \
   --adapter_path models/llama/ft/adapter \
   --device cpu \
   --provider CPUExecutionProvider \
   --use_ort_genai \
   --output_path models/llama/onnx \
   --log_level 1
```

Once the `olive auto-opt` command has successfully completed, you'll have:

1. The base model in an optimized ONNX format.
2. The adapter weights in a format for ONNX Runtime.

Olive and the ONNX runtime support the *multi-LoRA* model serving pattern, which greatly reduces the compute footprint of serving many adapters:


```{figure} ../../images/multi-lora-diagram.png
:width: 700px
:align: center

Multi-LoRA serving versus single-LoRA serving
```

### Inference model using ONNX Runtime

:::: {tab-set}

::: {tab-item} Python

Copy-and-paste the code below into a new Python file called `app.py`:

```python
import onnxruntime_genai as og

print("loading model and adapters...", end="", flush=True)
model = og.Model("models/llama/onnx/model")
adapters = og.Adapters(model)
adapters.load("models/llama/onnx/model/adapter_weights.onnx_adapter", "phrase_classifier")
print("DONE!")

tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

params = og.GeneratorParams(model)
params.set_search_options(max_length=100, past_present_share_buffer=False)

generator = og.Generator(model, params)
generator.set_active_adapter(adapters, "phrase_classifier")

user_input = "cricket is a wonderful sport"
generator.append_tokens(
    tokenizer.encode(f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
)

print(f"{user_input}")

while not generator.is_done():
    generator.generate_next_token()

    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)

print("\n")
```

Run the code with:

```bash
python app.py
```

:::

::::

