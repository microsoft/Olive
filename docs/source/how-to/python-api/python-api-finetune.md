# Fine-tuning Python API

The `finetune()` function provides programmatic access to fine-tuning functionality, allowing you to create LoRA/QLoRA adapters for models directly in Python.

## {octicon}`zap` Quickstart

```python
from olive import finetune

# Fine-tune a model with LoRA
workflow_output = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./finetuned_phi3",
    method="lora",
    data_name="squad",
    num_train_epochs=3,
    per_device_train_batch_size=4
)

# Access the fine-tuned adapter
if workflow_output.has_output_model():
    adapter = workflow_output.get_best_candidate()
    print(f"Adapter saved at: {adapter.model_path}")
```

## Function Signature

```python
def finetune(
    model_path: str,
    *,
    output_path: str = "finetune-output",
    method: str = "lora",
    data_name: Optional[str] = None,
    data_config: Optional[str] = None,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    log_level: int = 1,
    # LoRA parameters
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    # Training parameters
    **training_args
) -> WorkflowOutput
```

## Parameters

### Required Parameters
- **`model_path`** *(str)*: Path to model or Hugging Face model ID

### Fine-tuning Configuration
- **`output_path`** *(str)*: Directory to save fine-tuned model (default: "finetune-output")
- **`method`** *(str)*: Fine-tuning method - "lora" or "qlora" (default: "lora")
- **`data_name`** *(str)*: Dataset name (e.g., "squad", "gsm8k")
- **`data_config`** *(str)*: Path to data configuration file
- **`trust_remote_code`** *(bool)*: Allow remote code execution (default: False)
- **`torch_dtype`** *(str)*: PyTorch data type - "auto", "float16", "bfloat16" (default: "auto")

### LoRA Parameters
- **`lora_r`** *(int)*: LoRA rank (default: 16)
- **`lora_alpha`** *(int)*: LoRA alpha parameter (default: 32)
- **`lora_dropout`** *(float)*: LoRA dropout rate (default: 0.1)

### Training Parameters
You can pass any HuggingFace `TrainingArguments` as keyword arguments:
- **`num_train_epochs`** *(int)*: Number of training epochs
- **`per_device_train_batch_size`** *(int)*: Training batch size per device
- **`per_device_eval_batch_size`** *(int)*: Evaluation batch size per device
- **`learning_rate`** *(float)*: Learning rate
- **`warmup_steps`** *(int)*: Number of warmup steps
- **`logging_steps`** *(int)*: Logging frequency
- **`save_steps`** *(int)*: Model saving frequency
- **`eval_steps`** *(int)*: Evaluation frequency

## Return Value

Returns a `WorkflowOutput` object containing the fine-tuned adapter model and training metrics.

## Usage Examples

### Basic LoRA Fine-tuning

```python
from olive import finetune

# Fine-tune with default LoRA settings
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./adapters/phi3-squad",
    data_name="squad",
    num_train_epochs=3
)

if result.has_output_model():
    adapter = result.get_best_candidate()
    print(f"LoRA adapter: {adapter.model_path}")
```

### Advanced LoRA Configuration

```python
from olive import finetune

# Custom LoRA parameters and training settings
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./adapters/phi3-custom",
    method="lora",
    data_name="squad",
    # LoRA configuration
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    # Training configuration
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    trust_remote_code=True
)
```

### QLoRA Fine-tuning

```python
from olive import finetune

# Use QLoRA for memory-efficient fine-tuning
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./adapters/phi3-qlora",
    method="qlora",
    data_name="gsm8k",
    torch_dtype="bfloat16",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    learning_rate=1e-4
)
```

### Using Custom Dataset

```python
from olive import finetune

# Fine-tune with custom data configuration
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./adapters/phi3-custom-data",
    data_config="./my_data_config.json",
    method="lora",
    num_train_epochs=3,
    per_device_train_batch_size=4
)
```

## Working with Results

```python
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    data_name="squad",
    num_train_epochs=3
)

if result.has_output_model():
    adapter = result.get_best_candidate()

    # Access adapter information
    print(f"Adapter path: {adapter.model_path}")
    print(f"Training metrics: {adapter.metrics_value}")

    # Training metrics may include:
    # - train_loss, eval_loss
    # - train_accuracy, eval_accuracy
    # - train_runtime, eval_runtime
    metrics = adapter.metrics_value
    if 'eval_loss' in metrics:
        print(f"Final evaluation loss: {metrics['eval_loss']}")
```

## Integration with Other APIs

Fine-tuned adapters can be used with other Olive functions:

```python
from olive import finetune, extract_adapters

# Fine-tune model
adapter_result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    data_name="squad",
    num_train_epochs=3
)

# Extract the adapter for deployment
if adapter_result.has_output_model():
    adapter_model = adapter_result.get_best_candidate()

    extract_adapters(
        model_path=adapter_model.model_path,
        output_path="./extracted_adapters",
        format="onnx_adapter"
    )
```

## Data Configuration

You can specify datasets in several ways:

### Built-in Datasets
```python
# Use built-in dataset names
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    data_name="squad",  # or "gsm8k", "dolly", etc.
    num_train_epochs=3
)
```

### Custom Data Configuration
Create a JSON configuration file for custom datasets:

```json
{
    "data_name": "my_custom_dataset",
    "data_files": {
        "train": "./train.jsonl",
        "validation": "./val.jsonl"
    },
    "text_template": "### Question: {input}\n### Answer: {output}",
    "max_samples": 1000
}
```

```python
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    data_config="./my_data_config.json",
    num_train_epochs=3
)
```

## Equivalent CLI Command

```bash
# CLI equivalent
olive finetune \
    --model_name_or_path microsoft/phi-3-mini-4k-instruct \
    --output_path ./finetuned_phi3 \
    --method lora \
    --data_name squad \
    --num_train_epochs 3
```

```python
# Python API equivalent
from olive import finetune

result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./finetuned_phi3",
    method="lora",
    data_name="squad",
    num_train_epochs=3
)
```
