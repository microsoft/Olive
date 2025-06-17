# Python Interface

This document describes the Python interface of Olive, focusing on how to run optimization workflows and access their results.

## Running a Workflow with Python

Olive workflows can be executed programmatically using the Python API. All functions are available directly from the `olive` package.

### `run`

This is the most generic way to run an Olive workflow from a configuration file.

**Arguments:**
- `config` (Union[str, dict]): Path to config file or config dictionary.
- `input_model` (dict, optional): Input model configuration to override config file.
- `output_path` (str, optional): Output directory path.
- `log_level` (int, optional): Logging level (1-5).
- `setup` (bool): Setup environment needed to run the workflow. Defaults to `False`.
- `packages` (bool): List packages required to run the workflow. Defaults to `False`.
- `tempdir` (str, optional): Root directory for tempfile directories and files.
- `package_config` (str, optional): Path to optional package config file.

```python
from olive import run

# Run workflow from a configuration file
workflow_output = run("config.json")
```

The rest of the functions are specialized workflows for common tasks.

### `auto_opt`

Automatically optimize a model for performance.

**Arguments:**
- `model_path` (str): Path to model (file path or HuggingFace model name).
- `output_path` (str): Output directory path. Defaults to `"auto-opt-output"`.
- `device` (str): Target device ("cpu", "gpu", "npu"). Defaults to `"cpu"`.
- `provider` (str): Execution provider. Defaults to `"CPUExecutionProvider"`.
- `precision` (str): Output precision (fp32, fp16, int8, int4, nf4). Defaults to `fp32`.
- `data_name` (str, optional): Dataset name for evaluation.
- `split` (str, optional): Dataset split.
- `subset` (str, optional): Dataset subset.
- `input_cols` (list[str], optional): Input column names.
- `batch_size` (int): Batch size for evaluation. Defaults to `1`.
- `task` (str, optional): Model task (for HuggingFace models).
- `adapter_path` (str, optional): Path to adapter weights.
- `use_dynamo_exporter` (bool): Use dynamo export API. Defaults to `False`.
- `use_model_builder` (bool): Use model builder pass. Defaults to `False`.
- `use_qdq_encoding` (bool): Use QDQ encoding for quantization. Defaults to `False`.
- `use_ort_genai` (bool): Use ORT GenAI. Defaults to `False`.
- `enable_search` (bool, optional): Enable search optimization.
- `log_level` (int): Logging level (1-5). Defaults to `3`.
- `**kwargs`: Additional arguments.

```python
from olive import auto_opt

workflow_output = auto_opt(model_path="path/to/model")
```

### `finetune`

Fine-tune a model using LoRA or QLoRA.

**Arguments:**
- `model_path` (str): Path to HuggingFace model.
- `output_path` (str): Output directory path. Defaults to `"finetuned-adapter"`.
- `method` (str): Fine-tuning method ("lora", "qlora"). Defaults to `"lora"`.
- `lora_r` (int): LoRA rank value. Defaults to `64`.
- `lora_alpha` (int): LoRA alpha value. Defaults to `16`.
- `target_modules` (str, optional): Target modules for LoRA (comma-separated).
- `torch_dtype` (str): PyTorch dtype for training. Defaults to `"bfloat16"`.
- `data_name` (str, optional): Dataset name.
- `data_files` (str, optional): Path to data files.
- `text_template` (str, optional): Text template for formatting.
- `eval_split` (str, optional): Evaluation dataset split.
- `log_level` (int): Logging level (1-5). Defaults to `3`.
- `**training_kwargs`: HuggingFace training arguments.

```python
from olive import finetune

workflow_output = finetune(model_path="hf_model_name")
```

### `quantize`

Quantize a PyTorch or ONNX model.

**Arguments:**
- `model_path` (str): Path to model file.
- `output_path` (str): Output directory path. Defaults to `"quantized-model"`.
- `algorithm` (str): Quantization algorithm (e.g., "rtn", "gptq", "awq"). Defaults to `"rtn"`.
- `precision` (str): Quantization precision (int8, int4, etc.). Defaults to `"int8"`.
- `act_precision` (str): Activation precision for static quantization. Defaults to `"int8"`.
- `implementation` (str, optional): Specific implementation of quantization algorithms.
- `use_qdq_encoding` (bool): Use QDQ encoding in ONNX model. Defaults to `False`.
- `data_name` (str, optional): Dataset name (for static quantization).
- `log_level` (int): Logging level (1-5). Defaults to `3`.
- `**kwargs`: Additional quantization parameters.

```python
from olive import quantize

workflow_output = quantize(model_path="path/to/model")
```

### `capture_onnx_graph`

Capture ONNX graph for a PyTorch model.

**Arguments**:
- `model_path` (str): Path to PyTorch model or script.
- `output_path` (str): Output directory path. Defaults to `"captured-model"`.
- `log_level` (int): Logging level (1-5). Defaults to `3`.
- `**kwargs`: Additional arguments for `CaptureOnnxGraphCommand`.

```python
from olive import capture_onnx_graph

workflow_output = capture_onnx_graph(model_path="path/to/model")
```

### `generate_adapter`

Generate adapter for an ONNX model.

**Arguments**:
- `model_path` (str): Path to ONNX model.
- `output_path` (str): Output directory path. Defaults to `"generated-adapter"`.
- `adapter_format` (str): Format to save weights in. Defaults to `"onnx_adapter"`.
- `log_level` (int): Logging level (1-5). Defaults to `3`.
- `**kwargs`: Additional generation parameters.

```python
from olive import generate_adapter

workflow_output = generate_adapter(model_path="path/to/onnx/model")
```

### `session_params_tuning`

Tune ONNX Runtime session parameters for optimal performance.

**Arguments**:
- `model_path` (str): Path to ONNX model.
- `output_path` (str): Output directory path. Defaults to `"tuned-params"`.
- `device` (str): Target device. Defaults to `"cpu"`.
- `provider` (str): Execution provider. Defaults to `"CPUExecutionProvider"`.
- `cpu_cores` (int, optional): CPU cores for thread tuning.
- `io_bind` (bool): Enable IOBinding search. Defaults to `False`.
- `enable_cuda_graph` (bool): Enable CUDA graph. Defaults to `False`.
- `log_level` (int): Logging level (1-5). Defaults to `3`.
- `**kwargs`: Additional tuning parameters.

```python
from olive import session_params_tuning

workflow_output = session_params_tuning(model_path="path/to/onnx/model")
```

### Utility Functions

There are also utility functions that don't produce a `WorkflowOutput`:

- **`convert_adapters`**: Convert LoRA adapter weights to a format consumable by ONNX models.
  - **Arguments**: `adapter_path` (str), `output_path` (str), `adapter_format` (str), `dtype` (str), `log_level` (int), `**kwargs`.
- **`extract_adapters`**: Extract LoRA adapters from PyTorch model to separate files.
  - **Arguments**: `model_path` (str), `output_path` (str), `adapter_format` (str), `dtype` (str), `cache_dir` (str, optional), `log_level` (int), `**kwargs`.
- **`generate_cost_model`**: Generate a cost model for model splitting (HuggingFace models only).
  - **Arguments**: `model_path` (str), `output_path` (str), `log_level` (int), `**kwargs`.

## Accessing Optimization Results

Most API functions return a `WorkflowOutput` object. You can use it to access the optimized models and their metrics.

```python
from olive import run, WorkflowOutput

workflow_output: WorkflowOutput = run("config.json")

# Check if optimization produced any results
if workflow_output.has_output_model():
    # Get the best model overall
    best_model = workflow_output.get_best_candidate()
    print(f"Model path: {best_model.model_path}")
    print(f"Metrics: {best_model.metrics_value}")
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
