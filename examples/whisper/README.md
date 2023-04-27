# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-base) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*
- GPU: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Mixed-precision Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*

Outputs the best metrics, model, and corresponding Olive config.

**Note**: The template is not complete yet. There is no evaluator or search. Mixed precision, beam search, pre/post ops and tuning are not present.

## Prerequisites
### Prepare workflow config json
```
python prepare_config.py [--model_name MODEL_NAME] [--device {cpu,gpu}]
```

`model_name` is the name or path of the whisper model. The default value is `openai/whisper-base.en`.

`device` is one of `cpu` or `gpu`. It is the device that you want to optimize the model for. The default value is `cpu`.

## To optimize Whisper model run the sample config
First, install required packages according to passes.
```
python -m olive.workflows.run --config whisper_{device}_config.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config whisper_{device}_config.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
device = "cpu" # or "gpu"
olive_run(f"whisper_{device}_config.json")
```
