# Quantization Python API

The `quantize()` function provides programmatic access to model quantization, allowing you to reduce model size and improve inference speed through various quantization techniques.

## {octicon}`zap` Quickstart

```python
from olive import quantize

# Quantize an ONNX model
workflow_output = quantize(
    model_path="./model.onnx",
    output_path="./quantized_model",
    algorithm="rtn",
    precision="int8"
)

# Access the quantized model
if workflow_output.has_output_model():
    quantized_model = workflow_output.get_best_candidate()
    print(f"Quantized model: {quantized_model.model_path}")
    print(f"Model size reduction: {quantized_model.metrics_value}")
```

## Function Signature

```python
def quantize(
    model_path: str,
    *,
    output_path: str = "quantize-output",
    algorithm: str = "rtn",
    precision: str = "int8",
    data_name: Optional[str] = None,
    data_config: Optional[str] = None,
    providers: Optional[List[str]] = None,
    log_level: int = 1,
    **kwargs
) -> WorkflowOutput
```

## Parameters

### Required Parameters
- **`model_path`** *(str)*: Path to the model to quantize (ONNX format)

### Quantization Configuration
- **`output_path`** *(str)*: Directory to save quantized model (default: "quantize-output")
- **`algorithm`** *(str)*: Quantization algorithm - "rtn", "awq", "gptq", "smoothquant" (default: "rtn")
- **`precision`** *(str)*: Target precision - "int8", "int4" (default: "int8")
- **`data_name`** *(str)*: Dataset name for calibration (required for some algorithms)
- **`data_config`** *(str)*: Path to data configuration file
- **`providers`** *(List[str])*: ONNX Runtime execution providers
- **`log_level`** *(int)*: Logging verbosity level 1-5 (default: 1)

### Algorithm-Specific Parameters
Additional parameters can be passed for specific quantization algorithms:
- **`block_size`** *(int)*: Block size for block-wise quantization
- **`group_size`** *(int)*: Group size for group quantization
- **`calibration_samples`** *(int)*: Number of calibration samples

## Return Value

Returns a `WorkflowOutput` object containing the quantized model and quantization metrics.

## Usage Examples

### Basic Round-to-Nearest (RTN) Quantization

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

### AWQ Quantization

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

### GPTQ Quantization

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

### SmoothQuant

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

### Custom Data Configuration

```python
from olive import quantize

# Use custom calibration data
result = quantize(
    model_path="./model.onnx",
    output_path="./quantized_custom",
    algorithm="awq",
    precision="int4",
    data_config="./calibration_data_config.json"
)
```

## Working with Results

```python
result = quantize(
    model_path="./model.onnx",
    algorithm="awq",
    precision="int4"
)

if result.has_output_model():
    quantized = result.get_best_candidate()
    
    # Access quantization information
    print(f"Quantized model path: {quantized.model_path}")
    print(f"Quantization metrics: {quantized.metrics_value}")
    
    # Metrics may include:
    # - model_size_mb: Model size in megabytes
    # - accuracy: Model accuracy after quantization
    # - latency_ms: Inference latency
    # - compression_ratio: Size reduction ratio
    
    metrics = quantized.metrics_value
    if 'model_size_mb' in metrics:
        print(f"Model size: {metrics['model_size_mb']} MB")
    if 'compression_ratio' in metrics:
        print(f"Size reduction: {metrics['compression_ratio']:.2f}x")
```

## Integration with Other APIs

Quantization works well with other Olive APIs:

```python
from olive import auto_opt, quantize

# First optimize to ONNX
optimized = auto_opt(
    model_path="microsoft/phi-3-mini-4k-instruct",
    output_path="./optimized",
    device="cpu"
)

# Then quantize the optimized model
if optimized.has_output_model():
    onnx_model = optimized.get_best_candidate()
    
    quantized = quantize(
        model_path=onnx_model.model_path,
        output_path="./quantized",
        algorithm="awq",
        precision="int4",
        data_name="squad"
    )
```

## Quantization Algorithms

### Round-to-Nearest (RTN)
- **Best for**: Quick quantization without calibration data
- **Precision**: int8, int4
- **Calibration**: Not required

```python
result = quantize(
    model_path="./model.onnx",
    algorithm="rtn",
    precision="int8"
)
```

### AWQ (Activation-aware Weight Quantization)
- **Best for**: Language models with good accuracy preservation
- **Precision**: int4, int8
- **Calibration**: Required

```python
result = quantize(
    model_path="./model.onnx",
    algorithm="awq",
    precision="int4",
    data_name="squad",
    group_size=128
)
```

### GPTQ (Gradient-based Post-training Quantization)
- **Best for**: Large language models
- **Precision**: int4, int8
- **Calibration**: Required

```python
result = quantize(
    model_path="./model.onnx",
    algorithm="gptq", 
    precision="int4",
    data_name="pile",
    block_size=128
)
```

### SmoothQuant
- **Best for**: Models requiring high accuracy
- **Precision**: int8
- **Calibration**: Required

```python
result = quantize(
    model_path="./model.onnx",
    algorithm="smoothquant",
    precision="int8",
    data_name="squad"
)
```

## Data Configuration for Calibration

Some quantization algorithms require calibration data:

### Built-in Datasets
```python
result = quantize(
    model_path="./model.onnx",
    algorithm="awq",
    data_name="squad",  # or "pile", "c4", etc.
    calibration_samples=512
)
```

### Custom Calibration Data
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

## Equivalent CLI Command

```bash
# CLI equivalent
olive quantize \
    --model_path ./model.onnx \
    --output_path ./quantized_model \
    --algorithm awq \
    --precision int4 \
    --data_name squad
```

```python
# Python API equivalent
from olive import quantize

result = quantize(
    model_path="./model.onnx",
    output_path="./quantized_model",
    algorithm="awq",
    precision="int4",
    data_name="squad"
)
```