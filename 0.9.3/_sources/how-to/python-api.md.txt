# How to Use Python Interface

Olive provides Python API to transform models. See [Python API Reference](../reference/python_api.md) for list of supported APIs.

```python
from olive import run, WorkflowOutput

workflow_output: WorkflowOutput = run("config.json")

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

## Accessing Metrics and Configuration

```python
# Access metrics for input model
input_metrics = workflow_output.get_input_model_metrics()

# Get all available devices
devices = workflow_output.get_available_devices()

# Get model list from specific device and execution provider
output_models = workflow_output.CPU["CPUExecutionProvider"]
```

# Practical Tips

- Use `get_best_candidate()` to quickly obtain the model with the best metrics across all devices
- Use `get_output_models()` to get a list of all optimized models sorted by metrics
- Access device-specific outputs using the device value as a key or attribute: `workflow_output["cpu"]` or `workflow_output.cpu`. Device keys are case insensitive.
- Always verify the existence of specific devices or execution providers before accessing them, since the output model might be empty if a pass run has failed.
- Compare the input model metrics with optimized model metrics to quantify improvements
