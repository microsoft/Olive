# Python API

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

## Installation

The API is included with Olive:

```bash
pip install olive-ai
```

Import functions directly:

```python
from olive import auto_opt, finetune, quantize, run
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
| `capture_onnx_graph()` | Capture ONNX graphs from PyTorch | `olive capture-onnx-graph` |
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
    enable_search=True
)
```

```bash
# CLI equivalent - string arguments
olive auto-opt \
    --model_name_or_path microsoft/phi-3-mini-4k-instruct \
    --precision int8 \
    --enable_search
```
 
## Output Class Hierarchy

Olive organizes optimization results in a hierarchical structure:

- `WorkflowOutput`: Top-level container for all results across devices
  - Contains multiple `DeviceOutput` instances (one per device)
    - Each `DeviceOutput` contains multiple `ModelOutput` instances

### Working with WorkflowOutput

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

### ModelOutput Properties

Each `ModelOutput` instance contains:

- **`model_path`**: Path to the optimized model
- **`config`**: Configuration used to create the model
- **`metrics_value`**: Dictionary of evaluation metrics
- **`model_id`**: Unique identifier for the model
- **`model_type`**: Type of the model (e.g., "onnxmodel")
- **`model_config`**: Configuration details for the model

#### Key Methods

- `from_device()` - Get the device this model was optimized for
- `from_execution_provider()` - Get the execution provider this model was optimized for
- `from_pass()` - Get the Olive optimization pass that generated this model
- `get_parent_model_id()` - Get the ID of the parent model this was derived from
- `use_ort_extension()` - Check if the model uses the ONNXRuntime extension
- `get_inference_config()` - Get the model's inference configuration

```python
model = result.get_best_candidate()

# Model information
print(f"Path: {model.model_path}")
print(f"ID: {model.model_id}")
print(f"Device: {model.from_device()}")
print(f"Execution provider: {model.from_execution_provider()}")

# Metrics (varies by workflow)
metrics = model.metrics_value
if 'accuracy' in metrics:
    print(f"Accuracy: {metrics['accuracy']}")
if 'latency' in metrics:
    print(f"Latency: {metrics['latency']} ms")
if 'model_size_mb' in metrics:
    print(f"Size: {metrics['model_size_mb']} MB")
```

## Auto Optimization API

The `auto_opt()` function provides programmatic access to the same functionality as the `olive auto-opt` CLI command, automatically optimizing PyTorch/Hugging Face models into the ONNX format for efficient inference.

### Function Signature

```python
def auto_opt(
    model_path: str,
    *,
    output_path: str = "auto-opt-output",
    device: str = "cpu",
    provider: str = "CPUExecutionProvider",
    precision: Union[str, Precision] = "fp32",
    # Dataset options
    data_name: Optional[str] = None,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    input_cols: Optional[list[str]] = None,
    batch_size: int = 1,
    # Model options
    task: Optional[str] = None,
    adapter_path: Optional[str] = None,
    use_dynamo_exporter: bool = False,
    use_model_builder: bool = False,
    use_qdq_encoding: bool = False,
    use_ort_genai: bool = False,
    # Advanced options
    enable_search: Optional[bool] = None,
    max_iter: Optional[int] = None,
    log_level: int = 3,
    **kwargs,
) -> WorkflowOutput
```

### Parameters

#### Required Parameters
- **`model_path`** *(str)*: Path to model or Hugging Face model ID (e.g., "microsoft/phi-3-mini-4k-instruct")

#### Optional Parameters
- **`output_path`** *(str)*: Directory to save optimized model (default: "auto-opt-output")
- **`device`** *(str)*: Target device - "cpu", "gpu", or "npu" (default: "cpu")
- **`provider`** *(str)*: Execution provider - "CPUExecutionProvider", "CUDAExecutionProvider", etc.
- **`precision`** *(str)*: Model precision - "fp32", "fp16", "int8", "int4", or "nf4" (default: "fp32")
- **`data_name`** *(str)*: Dataset name for evaluation (e.g., "squad", "glue")
- **`split`** *(str)*: Dataset split to use for evaluation.
- **`subset`** *(str)*: Dataset subset to use for evaluation.
- **`input_cols`** *(List[str])*: Input columns to use from the dataset.
- **`batch_size`** *(int)*: Batch size for evaluation (default: 1).
- **`task`** *(str)*: Model task for Hugging Face models (e.g., "text-classification").
- **`adapter_path`** *(str)*: Path to LoRA adapter weights.
- **`use_dynamo_exporter`** *(bool)*: Use `torch.onnx.dynamo_export` for conversion (default: False).
- **`use_model_builder`** *(bool)*: Use ModelBuilder for optimization (default: False).
- **`use_qdq_encoding`** *(bool)*: Use QDQ format for quantized operators (default: False).
- **`use_ort_genai`** *(bool)*: Create ONNX Runtime Generate API configuration (default: False).
- **`enable_search`** *(bool)*: Enable optimization search across multiple configurations (default: False).
- **`max_iter`** *(int)*: Maximum search iterations when search is enabled.
- **`log_level`** *(int)*: Logging verbosity level 1-5 (default: 3).

### Usage Examples

#### Basic Optimization

```python
from olive import auto_opt

# Optimize a model with default settings
result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./models/phi3-optimized"
)

if result.has_output_model():
    model = result.get_best_candidate()
    print(f"Model path: {model.model_path}")
```

#### Advanced Optimization with Search

```python
from olive import auto_opt

# Use optimization search to find best configuration
result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./models/phi3-advanced",
    device="gpu",
    provider="CUDAExecutionProvider",
    precision="int4",
    data_name="squad",  # For evaluation
    enable_search=True,
    max_iter=10,
    trust_remote_code=True
)

# Compare all optimization candidates
for i, model in enumerate(result.model_outputs):
    print(f"Candidate {i}: {model.model_path}")
    print(f"Metrics: {model.metrics_value}")
```

#### For ONNX Runtime Generate API

```python
from olive import auto_opt

# Optimize for ONNX Runtime Generate API
result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./models/phi3-genai",
    precision="int4",
    use_ort_genai=True
)

# The output will include genai_config.json for easy inference
```

## ONNX Graph Capture API

The `capture_onnx_graph()` function provides programmatic access to capture the ONNX graph from a PyTorch model.

### Function Signature

```python
def capture_onnx_graph(
    model_path: str,
    *,
    output_path: str = "captured-model",
    log_level: int = 3,
    **kwargs
) -> WorkflowOutput:
```

### Parameters

- **`model_path`** *(str)*: Path to the PyTorch model or script to capture.
- **`output_path`** *(str)*: Directory to save the captured ONNX model (default: "captured-model").
- **`log_level`** *(int)*: Logging verbosity level 1-5 (default: 3).
- **`**kwargs`**: Additional arguments to pass to the capture command.

### Usage Example

```python
from olive import capture_onnx_graph

result = capture_onnx_graph(
    model_path="./my_pytorch_model.py",
    output_path="./onnx_models/captured"
)

if result.has_output_model():
    captured_model = result.get_best_candidate()
    print(f"Captured ONNX model saved at: {captured_model.model_path}")
```

## Fine-tuning API

The `finetune()` function provides programmatic access to fine-tuning functionality, allowing you to create LoRA/QLoRA adapters for models directly in Python.

### Function Signature

```python
def finetune(
    model_path: str,
    *,
    output_path: str = "finetune-output",
    method: str = "lora",
    # Dataset options
    data_name: Optional[str] = None,
    data_files: Optional[str] = None,
    text_template: Optional[str] = None,
    eval_split: Optional[str] = None,
    # LoRA parameters
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[str] = None,
    # Training parameters
    torch_dtype: str = "auto",
    log_level: int = 1,
    **training_args
) -> WorkflowOutput
```

### Parameters

#### Required Parameters
- **`model_path`** *(str)*: Path to model or Hugging Face model ID

#### Fine-tuning Configuration
- **`output_path`** *(str)*: Directory to save fine-tuned model (default: "finetune-output")
- **`method`** *(str)*: Fine-tuning method - "lora" or "qlora" (default: "lora")
- **`data_name`** *(str)*: Dataset name (e.g., "squad", "gsm8k")
- **`data_files`** *(str)*: Path to data files.
- **`text_template`** *(str)*: Template for formatting text data.
- **`eval_split`** *(str)*: Dataset split for evaluation.
- **`torch_dtype`** *(str)*: PyTorch data type - "auto", "float16", "bfloat16" (default: "auto")

#### LoRA Parameters
- **`lora_r`** *(int)*: LoRA rank (default: 16)
- **`lora_alpha`** *(int)*: LoRA alpha parameter (default: 32)
- **`target_modules`** *(str)*: Comma-separated list of module names to apply LoRA to.

#### Training Parameters
You can pass any HuggingFace `TrainingArguments` as keyword arguments. Common arguments include:
- **`num_train_epochs`** *(int)*: Number of training epochs
- **`per_device_train_batch_size`** *(int)*: Training batch size per device
- **`per_device_eval_batch_size`** *(int)*: Evaluation batch size per device
- **`learning_rate`** *(float)*: Learning rate
- **`warmup_steps`** *(int)*: Number of warmup steps
- **`logging_steps`** *(int)*: Logging frequency
- **`save_steps`** *(int)*: Model saving frequency
- **`eval_steps`** *(int)*: Evaluation frequency

### Usage Examples

#### Basic LoRA Fine-tuning

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

#### Advanced LoRA Configuration

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
    # Training configuration
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    lora_dropout=0.05,
    warmup_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    trust_remote_code=True
)
```

#### QLoRA Fine-tuning

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

### Data Configuration

You can specify datasets in several ways:

#### Built-in Datasets
```python
# Use built-in dataset names
result = finetune(
    model_path="microsoft/phi-3-mini-4k-instruct",
    data_name="squad",  # or "gsm8k", "dolly", etc.
    num_train_epochs=3
)
```

#### Custom Data Configuration
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

## Quantization API

The `quantize()` function provides programmatic access to model quantization, allowing you to reduce model size and improve inference speed through various quantization techniques.

### Function Signature

```python
def quantize(
    model_path: str,
    *,
    output_path: str = "quantize-output",
    algorithm: str = "rtn",
    precision: str = "int8",
    act_precision: str = "int8",
    implementation: Optional[str] = None,
    use_qdq_encoding: bool = False,
    data_name: Optional[str] = None,
    data_config: Optional[str] = None,
    log_level: int = 1,
    **kwargs
) -> WorkflowOutput
```

### Parameters

#### Required Parameters
- **`model_path`** *(str)*: Path to the model to quantize (ONNX format)

#### Quantization Configuration
- **`output_path`** *(str)*: Directory to save quantized model (default: "quantize-output")
- **`algorithm`** *(str)*: Quantization algorithm - e.g., "rtn", "awq", "gptq", "smoothquant" (default: "rtn")
- **`precision`** *(str)*: Target precision for weights - e.g., "int8", "int4" (default: "int8")
- **`act_precision`** *(str)*: Target precision for activations in static quantization (default: "int8")
- **`implementation`** *(str)*: Specific implementation of the quantization algorithm to use.
- **`use_qdq_encoding`** *(bool)*: Use QDQ format for quantized operators (default: False).
- **`data_name`** *(str)*: Dataset name for calibration (required for some algorithms).
- **`data_config`** *(str)*: Path to data configuration file.
- **`log_level`** *(int)*: Logging verbosity level 1-5 (default: 1).

#### Algorithm-Specific Parameters
Additional parameters can be passed as keyword arguments (`**kwargs`) for specific quantization algorithms:
- **`block_size`** *(int)*: Block size for block-wise quantization
- **`group_size`** *(int)*: Group size for group quantization

### Usage Examples

#### Basic Round-to-Nearest (RTN) Quantization

```python
from olive import quantize

# Simple RTN quantization
result = quantize(
    model_path="./model.onnx",
    output_path="./quantized_rtn",
    algorithm="rtn",
    precision="int8"
)

if result.has_output_model():
    quantized = result.get_best_candidate()
    print(f"Quantized model: {quantized.model_path}")
```

#### AWQ Quantization

```python
from olive import quantize

# AWQ quantization with calibration data
result = quantize(
    model_path="./model.onnx",
    output_path="./quantized_awq",
    algorithm="awq",
    precision="int4",
    data_name="squad",
    group_size=128,
    calibration_samples=512
)
```

#### GPTQ Quantization

```python
from olive import quantize

# GPTQ quantization for language models
result = quantize(
    model_path="./model.onnx",
    output_path="./quantized_gptq",
    algorithm="gptq",
    precision="int4",
    data_name="pile",
    block_size=128,
    calibration_samples=256
)
```

#### SmoothQuant

```python
from olive import quantize

# SmoothQuant for better accuracy preservation
result = quantize(
    model_path="./model.onnx",
    output_path="./quantized_smooth",
    algorithm="smoothquant",
    precision="int8",
    data_name="squad",
    calibration_samples=100
)
```

### Quantization Algorithms

#### Round-to-Nearest (RTN)
- **Best for**: Quick quantization without calibration data
- **Precision**: int8, int4
- **Calibration**: Not required

#### AWQ (Activation-aware Weight Quantization)
- **Best for**: Language models with good accuracy preservation
- **Precision**: int4, int8
- **Calibration**: Required

#### GPTQ (Gradient-based Post-training Quantization)
- **Best for**: Large language models
- **Precision**: int4, int8
- **Calibration**: Required

#### SmoothQuant
- **Best for**: Models requiring high accuracy
- **Precision**: int8
- **Calibration**: Required

### Data Configuration for Calibration

Some quantization algorithms require calibration data:

#### Built-in Datasets
```python
result = quantize(
    model_path="./model.onnx",
    algorithm="awq",
    data_name="squad",  # or "pile", "c4", etc.
    calibration_samples=512
)
```

#### Custom Calibration Data
Create a JSON configuration file:

```json
{
    "data_name": "custom_calibration",
    "data_files": "./calibration_data.jsonl",
    "input_cols": ["input_text"],
    "max_samples": 1000,
    "preprocessing": {
        "max_length": 512,
        "truncation": true
    }
}
```

```python
result = quantize(
    model_path="./model.onnx",
    algorithm="awq",
    data_config="./calibration_config.json"
)
```

## Run Workflow API

The `run()` function provides programmatic access to execute Olive workflows from configuration files or dictionaries, giving you full control over complex optimization pipelines.

### Function Signature

```python
def run(
    config: Union[str, Dict[str, Any]],
    *,
    input_model: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    log_level: Optional[int] = None,
    setup: bool = False,
    packages: bool = False,
    tempdir: Optional[str] = None,
    package_config: Optional[str] = None,
) -> WorkflowOutput
```

### Parameters

#### Required Parameters
- **`config`** *(str | dict)*: Path to configuration file or configuration dictionary

#### Optional Parameters
- **`input_model`** *(dict)*: Input model configuration to override config file
- **`output_path`** *(str)*: Output directory path (overrides config)
- **`log_level`** *(int)*: Logging level 1-5 (overrides config)
- **`setup`** *(bool)*: Setup environment needed to run the workflow (default: False)
- **`packages`** *(bool)*: List packages required to run the workflow (default: False)
- **`tempdir`** *(str)*: Root directory for temporary files
- **`package_config`** *(str)*: Path to optional package configuration file

### Usage Examples

#### Running from Configuration File

```python
from olive import run

# Execute workflow from JSON file
result = run(
    config="./my_workflow.json",
    output_path="./results",
    log_level=2
)

# Process all workflow outputs
for i, model in enumerate(result.model_outputs):
    print(f"Model {i}: {model.model_path}")
    print(f"Metrics: {model.metrics_value}")
```

#### Running from Configuration Dictionary

```python
from olive import run

# Define workflow configuration as dictionary
config = {
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/phi-3-mini-4k-instruct",
        "trust_remote_code": True
    },
    "passes": {
        "convert": {
            "type": "OnnxConversion"
        },
        "optimize": {
            "type": "OrtTransformersOptimization"
        },
        "quantize": {
            "type": "OnnxQuantization",
            "config": {
                "quant_mode": "rtn",
                "weight_type": "QUInt8"
            }
        }
    },
    "evaluators": {
        "common_evaluator": {
            "metrics": [
                {"name": "accuracy", "type": "accuracy"},
                {"name": "latency", "type": "latency"}
            ]
        }
    },
    "search_strategy": {
        "execution_order": "joint",
        "search_algorithm": "tpe"
    }
}

result = run(
    config=config,
    output_path="./workflow_results"
)
```

#### Multi-Pass Optimization Workflow

Create a JSON configuration file `workflow.json`:

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "microsoft/phi-3-mini-4k-instruct"
    },
    "passes": {
        "conversion": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 17
            }
        },
        "optimization": {
            "type": "OrtTransformersOptimization",
            "config": {
                "optimization_level": "all"
            }
        },
        "quantization": {
            "type": "OnnxQuantization",
            "config": {
                "quant_mode": "rtn",
                "weight_type": "QUInt8"
            }
        }
    },
    "evaluators": {
        "accuracy_evaluator": {
            "metrics": [
                {
                    "name": "accuracy",
                    "type": "accuracy",
                    "data_config": {
                        "data_name": "squad",
                        "batch_size": 1
                    }
                }
            ]
        }
    },
    "engine": {
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "exhaustive"
        },
        "output_dir": "./workflow_output"
    }
}
```

```python
from olive import run

result = run(config="./workflow.json")
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

### 4. Complete Optimization Pipeline
```python
from olive import auto_opt, quantize, extract_adapters

# Step 1: Auto-optimize to ONNX
optimized = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./optimized",
    device="cpu"
)

# Step 2: Apply advanced quantization
if optimized.has_output_model():
    onnx_model = optimized.get_best_candidate()

    quantized = quantize(
        model_path=onnx_model.model_path,
        output_path="./quantized",
        algorithm="awq",
        precision="int4",
        data_name="squad"
    )

    # Step 3: Extract for deployment
    if quantized.has_output_model():
        final_model = quantized.get_best_candidate()
        extract_adapters(
            model_path=final_model.model_path,
            output_path="./deployment"
        )
```

## Practical Tips

- Use `get_best_candidate()` to quickly obtain the model with the best metrics across all devices
- Use `get_output_models()` to get a list of all optimized models sorted by metrics
- Access device-specific outputs using the device value as a key or attribute: `workflow_output["cpu"]` or `workflow_output.cpu`. Device keys are case insensitive.
- Always verify the existence of specific devices or execution providers before accessing them, since the output model might be empty if a pass run has failed.
- Compare the input model metrics with optimized model metrics to quantify improvements
- Chain different optimization techniques for maximum performance gains
- Use the `run()` function for complex multi-pass workflows that require custom configuration
``` 