# Python API Overview

The Olive Python API provides programmatic access to all optimization functionality, allowing you to execute workflows directly in Python code and receive structured `WorkflowOutput` objects with detailed results.

## {octicon}`zap` Quickstart

```python
from olive import auto_opt, finetune, quantize

# Auto-optimize a model
result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    device="cpu",
    precision="int8"
)

# Access the optimized model
if result.has_output_model():
    best_model = result.get_best_candidate()
    print(f"Model: {best_model.model_path}")
    print(f"Metrics: {best_model.metrics_value}")
```

## Available Functions

The Python API provides functions corresponding to all CLI commands:

### Model Optimization Functions
These functions execute workflows and return `WorkflowOutput` objects:

| Function | Description | CLI Equivalent |
|----------|-------------|----------------|
| `auto_opt()` | Auto-optimize models for performance | `olive auto-opt` |
| `finetune()` | Fine-tune models using LoRA/QLoRA | `olive finetune` |
| `quantize()` | Quantize models for reduced size | `olive quantize` |
| `capture_onnx()` | Capture ONNX graphs from PyTorch | `olive capture-onnx-graph` |
| `generate_adapter()` | Generate adapters for ONNX models | `olive generate-adapter` |
| `session_params_tuning()` | Tune ONNX Runtime parameters | `olive session-params-tuning` |
| `run()` | Execute workflows from configuration | `olive run` |

### Utility Functions
These functions perform operations and return `None`:

| Function | Description | CLI Equivalent |
|----------|-------------|----------------|
| `convert_adapters()` | Convert adapter formats | `olive convert-adapters` |
| `extract_adapters()` | Extract LoRA adapters | `olive extract-adapters` |
| `generate_cost_model()` | Generate cost models for splitting | `olive generate-cost-model` |

## Key Benefits

### 1. Structured Returns
Instead of file outputs, get structured objects with detailed information:

```python
from olive import auto_opt

result = auto_opt(model_path="microsoft/phi-3-mini-4k-instruct")

# Access structured results
if result.has_output_model():
    model = result.get_best_candidate()
    print(f"Path: {model.model_path}")
    print(f"Config: {model.config}")
    print(f"Metrics: {model.metrics_value}")
    
    # Iterate through all candidates
    for candidate in result.model_outputs:
        print(f"Candidate: {candidate.model_path}")
```

### 2. Python-native Parameters
Use Python data types instead of command-line strings:

```python
# Python API - native types
result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    precision="int8",
    enable_search=True,
    max_iter=10
)
```

```bash
# CLI equivalent - string arguments
olive auto-opt \
    --model_name_or_path microsoft/phi-3-mini-4k-instruct \
    --precision int8 \
    --enable_search \
    --max_iter 10
```

### 3. Integration and Chaining
Easily chain operations and integrate with other Python code:

```python
from olive import auto_opt, quantize, extract_adapters

# Chain operations
optimized = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./optimized" 
)

if optimized.has_output_model():
    model = optimized.get_best_candidate()
    
    # Further quantize
    quantized = quantize(
        model_path=model.model_path,
        algorithm="awq",
        precision="int4"
    )
    
    # Extract for deployment
    if quantized.has_output_model():
        final_model = quantized.get_best_candidate()
        extract_adapters(
            model_path=final_model.model_path,
            output_path="./deployment"
        )
```

## Working with WorkflowOutput

All optimization functions return `WorkflowOutput` objects:

```python
from olive import auto_opt

result = auto_opt(model_path="microsoft/phi-3-mini-4k-instruct")

# Check if workflow produced models
if result.has_output_model():
    # Get the best model (highest scoring)
    best = result.get_best_candidate()
    
    # Get all models
    all_models = result.model_outputs
    
    # Access model information
    for model in all_models:
        print(f"Model: {model.model_path}")
        print(f"Metrics: {model.metrics_value}")
        print(f"Config: {model.config}")
```

## ModelOutput Properties

Each `ModelOutput` instance contains:

- **`model_path`**: Path to the optimized model
- **`config`**: Configuration used to create the model
- **`metrics_value`**: Dictionary of evaluation metrics
- **`model_id`**: Unique identifier for the model

```python
model = result.get_best_candidate()

# Model information
print(f"Path: {model.model_path}")
print(f"ID: {model.model_id}")

# Metrics (varies by workflow)
metrics = model.metrics_value
if 'accuracy' in metrics:
    print(f"Accuracy: {metrics['accuracy']}")
if 'latency' in metrics:
    print(f"Latency: {metrics['latency']} ms")
if 'model_size_mb' in metrics:
    print(f"Size: {metrics['model_size_mb']} MB")
```

## Error Handling

The API provides proper Python exception handling:

```python
from olive import auto_opt

try:
    result = auto_opt(
        model_path="nonexistent/model",
        output_path="./output"
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Optimization failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration vs Parameters

You can use the API in two ways:

### 1. Function Parameters (Recommended)
```python
from olive import auto_opt

# Use function parameters
result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    device="cpu",
    precision="int8",
    data_name="squad"
)
```

### 2. Configuration Files/Dictionaries
```python
from olive import run

# Use configuration dictionary
config = {
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/phi-3-mini-4k-instruct"
    },
    "passes": {
        "convert": {"type": "OnnxConversion"},
        "quantize": {"type": "OnnxQuantization"}
    }
}

result = run(config=config)
```

## Common Patterns

### 1. Quick Optimization
```python
from olive import auto_opt

# One-line optimization
result = auto_opt("microsoft/phi-3-mini-4k-instruct", precision="int8")
```

### 2. Fine-tune then Optimize
```python
from olive import finetune, auto_opt

# Fine-tune first
adapter = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    data_name="squad",
    num_train_epochs=3
)

# Then optimize
if adapter.has_output_model():
    optimized = auto_opt(
        model_path=adapter.get_best_candidate().model_path,
        precision="int4"
    )
```

### 3. Batch Processing
```python
from olive import auto_opt

models = [
    "microsoft/phi-3-mini-4k-instruct",
    "microsoft/phi-3.5-mini-instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct"
]

results = []
for model_path in models:
    result = auto_opt(model_path=model_path, precision="int8")
    if result.has_output_model():
        results.append(result.get_best_candidate())

# Compare results
for model in results:
    print(f"Model: {model.model_path}")
    print(f"Metrics: {model.metrics_value}")
```

## Installation

The API is included with Olive:

```bash
pip install olive-ai
```

Import functions directly:

```python
from olive import auto_opt, finetune, quantize, run
```

Or import the full API:

```python
import olive.api as olive_api

result = olive_api.auto_opt(model_path="microsoft/phi-3-mini-4k-instruct")
```

## Next Steps

- **[Auto-optimization API](python-api-auto-opt.html)** - Detailed auto-opt documentation
- **[Fine-tuning API](python-api-finetune.html)** - LoRA/QLoRA fine-tuning
- **[Quantization API](python-api-quantize.html)** - Model quantization techniques  
- **[Workflow API](python-api-run.html)** - Advanced workflow execution