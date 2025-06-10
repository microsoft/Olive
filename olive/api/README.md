# Olive Python API

This directory contains the Python API for Olive, which provides programmatic access to all CLI commands and returns `WorkflowOutput` objects containing `ModelOutput` instances.

## Overview

The Olive Python API allows you to:

- Execute optimization workflows programmatically
- Access results as structured `WorkflowOutput` objects
- Retrieve individual `ModelOutput` instances with metrics and paths
- Use the same functionality as CLI commands but with Python function calls

## Main Functions

### Model Optimization Functions
These functions return `WorkflowOutput` objects containing optimized models:

- **`auto_opt()`** - Automatically optimize models for performance
- **`finetune()`** - Fine-tune models using LoRA/QLoRA
- **`quantize()`** - Quantize models for reduced size and faster inference
- **`capture_onnx()`** - Capture ONNX graphs from PyTorch models
- **`generate_adapter()`** - Generate adapters for ONNX models
- **`session_params_tuning()`** - Tune ONNX Runtime session parameters
- **`run()`** - Execute workflows from configuration files/dicts

### Utility Functions
These functions perform operations but don't return model outputs:

- **`configure_qualcomm_sdk()`** - Configure Qualcomm SDK
- **`convert_adapters()`** - Convert adapter formats
- **`extract_adapters()`** - Extract LoRA adapters from PyTorch models
- **`generate_cost_model()`** - Generate cost models for splitting
- **`manage_aml_compute()`** - Manage AzureML compute resources
- **`shared_cache()`** - Manage shared cache operations

## Quick Start

```python
from olive import auto_opt, finetune, quantize, run

# Auto-optimize a HuggingFace model
workflow_output = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./optimized",
    device="cpu",
    precision="int8"
)

# Access the best model
if workflow_output.has_output_model():
    best_model = workflow_output.get_best_candidate()
    print(f"Model: {best_model.model_path}")
    print(f"Metrics: {best_model.metrics_value}")

# Fine-tune with LoRA
adapter_output = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct", 
    data_name="squad",
    method="lora",
    num_train_epochs=3
)

# Run from config
config = {...}  # Your workflow config
result = run(config, output_path="./results")
```

## WorkflowOutput and ModelOutput

The API functions return `WorkflowOutput` objects that contain:

- **`ModelOutput`** instances representing optimized models
- Methods to access the best models, filter by device, get metrics, etc.
- Full traceability of the optimization process

Key methods on `WorkflowOutput`:
- `has_output_model()` - Check if optimization produced models
- `get_best_candidate()` - Get the best model overall
- `get_best_candidate_by_device(device)` - Get best model for specific device
- `get_output_models()` - Get all output models
- `get_output_models_by_device(device)` - Get models for specific device

Key attributes on `ModelOutput`:
- `model_path` - Path to the optimized model
- `model_type` - Type of model (e.g., "onnxmodel")
- `metrics_value` - Performance metrics
- `from_device()` - Device the model was optimized for
- `from_execution_provider()` - Execution provider used
- `from_pass()` - Optimization pass that generated this model

## Examples

See `examples/python_api_usage.py` for comprehensive usage examples.

## Comparison with CLI

The Python API provides the same functionality as CLI commands:

| CLI Command | Python Function | Returns |
|-------------|----------------|---------|
| `olive auto-opt` | `auto_opt()` | `WorkflowOutput` |
| `olive finetune` | `finetune()` | `WorkflowOutput` |
| `olive quantize` | `quantize()` | `WorkflowOutput` |
| `olive run` | `run()` | `WorkflowOutput` |
| `olive capture-onnx` | `capture_onnx()` | `WorkflowOutput` |
| `olive generate-adapter` | `generate_adapter()` | `WorkflowOutput` |
| `olive tune-session-params` | `session_params_tuning()` | `WorkflowOutput` |
| `olive configure-qualcomm-sdk` | `configure_qualcomm_sdk()` | `None` |
| `olive convert-adapters` | `convert_adapters()` | `None` |
| `olive extract-adapters` | `extract_adapters()` | `None` |
| `olive generate-cost-model` | `generate_cost_model()` | `None` |
| `olive manage-aml-compute` | `manage_aml_compute()` | `None` |
| `olive shared-cache` | `shared_cache()` | `None` |

## Installation

The Python API is included with Olive. Install Olive and import the functions:

```python
pip install olive-ai
from olive import auto_opt, finetune  # etc.
```