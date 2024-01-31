# phi2 optimization with Olive
This folder contains an example of phi2 optimization with Olive workflow.

- CPU: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

## Prerequisites
* einops

## Usage
```bash
python -m olive.workflows.run --config phi2_optimize.json
```

## Limitations
when https://github.com/huggingface/optimum/issues/1642 is fixed, we need turn on the post_process by changing `no_post_process` to False in the config file.
