# SqueezeNet Latency Optimization with DirectML
This folder contains a sample use case of Olive to optimize the [SqueezeNet](https://pytorch.org/hub/pytorch_vision_squeezenet/) model using ONNX conversion, conversion to FLOAT16, and general ONNX performance tuning.

Performs optimization pipeline:

    PyTorch Model -> [Convert to ONNX] -> [FP16 Conversion] -> [Tune performance] -> Optimized FP16 ONNX Model

Outputs the best metrics, model, and corresponding Olive config.

## Optimize SqueezeNet
```
python -m olive.workflows.run --config squeezenet_config.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("squeezenet_config.json")
```
