# Auto Optimization Python API

The `auto_opt()` function provides programmatic access to the same functionality as the `olive auto-opt` CLI command, automatically optimizing PyTorch/Hugging Face models into the ONNX format for efficient inference.

## {octicon}`zap` Quickstart

You can use the Python API to optimize models directly in your Python code:

```python
from olive import auto_opt

# Auto-optimize a HuggingFace model
workflow_output = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./optimized_phi3",
    device="cpu",
    provider="CPUExecutionProvider",
    precision="int8"
)

# Access the optimized model
if workflow_output.has_output_model():
    best_model = workflow_output.get_best_candidate()
    print(f"Optimized model saved at: {best_model.model_path}")
    print(f"Model metrics: {best_model.metrics_value}")
```

## Function Signature

```python
def auto_opt(
    model_path: str,
    *,
    output_path: str = "auto-opt-output",
    device: str = "cpu",
    provider: Optional[str] = None,
    precision: str = "fp32",
    optimization_config: Optional[str] = None,
    data_name: Optional[str] = None,
    data_config: Optional[str] = None,
    trust_remote_code: bool = False,
    use_ort_genai: bool = False,
    log_level: int = 1,
    enable_search: bool = False,
    max_iter: int = 50,
    **kwargs
) -> WorkflowOutput
```

## Parameters

### Required Parameters
- **`model_path`** *(str)*: Path to model or Hugging Face model ID (e.g., "microsoft/phi-3-mini-4k-instruct")

### Optional Parameters
- **`output_path`** *(str)*: Directory to save optimized model (default: "auto-opt-output")
- **`device`** *(str)*: Target device - "cpu", "gpu", or "npu" (default: "cpu")
- **`provider`** *(str)*: Execution provider - "CPUExecutionProvider", "CUDAExecutionProvider", etc.
- **`precision`** *(str)*: Model precision - "fp32", "fp16", "int8", or "int4" (default: "fp32")
- **`optimization_config`** *(str)*: Path to optimization configuration file
- **`data_name`** *(str)*: Dataset name for evaluation (e.g., "squad", "glue")
- **`data_config`** *(str)*: Path to data configuration file
- **`trust_remote_code`** *(bool)*: Allow remote code execution for HuggingFace models (default: False)
- **`use_ort_genai`** *(bool)*: Create ONNX Runtime Generate API configuration (default: False)
- **`log_level`** *(int)*: Logging verbosity level 1-5 (default: 1)
- **`enable_search`** *(bool)*: Enable optimization search across multiple configurations (default: False)
- **`max_iter`** *(int)*: Maximum search iterations (default: 50)

## Return Value

Returns a `WorkflowOutput` object containing:
- **Optimized models**: Access via `get_best_candidate()` or iterate through all candidates
- **Metrics**: Performance metrics for each optimized model
- **Metadata**: Configuration and optimization details

## Usage Examples

### Basic Optimization

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

### Advanced Optimization with Search

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

### For ONNX Runtime Generate API

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

## Working with Results

The `WorkflowOutput` object provides several methods to access optimization results:

```python
result = auto_opt(model_path="microsoft/phi-3-mini-4k-instruct")

# Check if optimization produced any models
if result.has_output_model():
    # Get the best performing model
    best_model = result.get_best_candidate()

    # Access model information
    print(f"Model path: {best_model.model_path}")
    print(f"Metrics: {best_model.metrics_value}")
    print(f"Config: {best_model.config}")

    # Iterate through all optimized models
    for model in result.model_outputs:
        print(f"Model: {model.model_path}")
        print(f"Accuracy: {model.metrics_value.get('accuracy', 'N/A')}")
        print(f"Latency: {model.metrics_value.get('latency', 'N/A')}")
```

## Integration with Other APIs

The `auto_opt()` function works well with other Olive Python APIs:

```python
from olive import auto_opt, quantize

# First auto-optimize
optimized = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./models/optimized"
)

# Then apply additional quantization
if optimized.has_output_model():
    best_model = optimized.get_best_candidate()

    quantized = quantize(
        model_path=best_model.model_path,
        output_path="./models/quantized",
        algorithm="awq",
        precision="int4"
    )
```

## Equivalent CLI Command

The Python API provides the same functionality as the CLI:

```bash
# CLI equivalent
olive auto-opt \
    --model_name_or_path microsoft/phi-3-mini-4k-instruct \
    --output_path ./optimized_phi3 \
    --device cpu \
    --provider CPUExecutionProvider \
    --precision int8
```

```python
# Python API equivalent
from olive import auto_opt

result = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./optimized_phi3",
    device="cpu",
    provider="CPUExecutionProvider",
    precision="int8"
)
```