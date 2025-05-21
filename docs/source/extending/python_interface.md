# Python Interface

This document describes the Python interface of Olive, focusing on how to run optimization workflows and access their results.

## Running a Workflow with Python

Olive workflows can be executed programmatically using the Python API. Start by learning about [how to configure workflows](https://microsoft.github.io/Olive/how-to/configure-workflows/build-workflow.html), then run your configuration:

```python
from olive.workflows import run as olive_run

# Run workflow from a configuration file
workflow_output = olive_run("config.json")
```

## Output Class Hierarchy

Olive organizes optimization results in a hierarchical structure:

- `WorkflowOutput`: Top-level container for all results across devices
  - Contains multiple `DeviceOutput` instances (one per device)
    - Each `DeviceOutput` contains multiple `ModelOutput` instances

## Output Classes in Detail

### WorkflowOutput

The `WorkflowOutput` class organizes results from an entire optimization workflow, containing outputs across different hardware devices and execution providers.

```python
from olive import WorkflowOutput
```

#### Key Methods

- `get_input_model_metrics()` - Get the metrics for the input model
- `get_available_devices()` - Get a list of devices that the workflow ran on
- `has_output_model()` - Check if any optimized models are available
- `get_output_models_by_device(device)` - Get all optimized models for a specific device
- `get_output_model_by_id(model_id)` - Get a specific model by its ID
- `get_output_models()` - Get all optimized models sorted by metrics
- `get_best_candidate_by_device(device)` - Get the best model for a specific device
- `get_best_candidate()` - Get the best model across all devices
- `trace_back_run_history(model_id)` - Get the optimization history for a specific model

### DeviceOutput

The `DeviceOutput` class groups model outputs for a specific device, containing results for different execution providers on that device.

```python
from olive import DeviceOutput
```

#### Key Attributes

- `device` - The device type (e.g., "cpu", "gpu")

#### Key Methods

- `has_output_model()` - Check if any model outputs are available for this device
- `get_output_models()` - Get all model outputs for this device
- `get_best_candidate()` - Get the best model output for this device based on metrics
- `get_best_candidate_by_execution_provider(execution_provider)` - Get the best model for a specific execution provider

### ModelOutput

The `ModelOutput` class represents an individual optimized model, containing its path, metrics, and configuration.

```python
from olive import ModelOutput
```

#### Key Attributes

- `metrics` - Dictionary containing the model's performance metrics
- `metrics_value` - Simplified version of metrics with just the values
- `model_path` - Path to the optimized model file
- `model_id` - Unique identifier for this model
- `model_type` - Type of the model (e.g., "onnxmodel")
- `model_config` - Configuration details for the model

#### Key Methods

- `from_device()` - Get the device this model was optimized for
- `from_execution_provider()` - Get the execution provider this model was optimized for
- `from_pass()` - Get the Olive optimization pass that generated this model
- `get_parent_model_id()` - Get the ID of the parent model this was derived from
- `use_ort_extension()` - Check if the model uses the ONNXRuntime extension
- `get_inference_config()` - Get the model's inference configuration


## Usage Examples

### Accessing Optimization Results

```python
from olive.workflows import run as olive_run
from olive import WorkflowOutput

workflow_output = olive_run("config.json")

# Check if optimization produced any results
if workflow_output.has_output_model():
    # Get the best model overall
    best_model = workflow_output.get_best_candidate()
    print(f"Model path: {best_model.model_path}")
    print(f"Model type: {best_model.model_type}")
    print(f"Device: {best_model.from_device()}")
    print(f"Execution provider: {best_model.from_execution_provider()}")
    print(f"Metrics: {best_model.metrics_value}")

    # Get the best model for CPU
    best_cpu_model = workflow_output.get_best_candidate_by_device("CPU")

    # Get all models for GPU
    gpu_models = workflow_output.get_output_models_by_device("GPU")
```

### Accessing Metrics and Configuration

```python
# Access metrics for input model
input_metrics = workflow_output.get_input_model_metrics()

# Get all available devices
devices = workflow_output.get_available_devices()

# Get model list from specific device and execution provider
output_models = workflow_output.CPU["CPUExecutionProvider"]
```

## Practical Tips

- Use `get_best_candidate()` to quickly obtain the model with the best metrics across all devices
- Use `get_output_models()` to get a list of all optimized models sorted by metrics
- Access device-specific outputs using the device value as a key or attribute: `workflow_output["cpu"]` or `workflow_output.cpu`. Device keys are case insensitive.
- Always verify the existence of specific devices or execution providers before accessing them, since the output model might be empty if a pass run has failed.
- Compare the input model metrics with optimized model metrics to quantify improvements
