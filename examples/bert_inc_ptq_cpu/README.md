# BERT optimization with Intel® Neural Compressor PTQ on CPU
This folder contains a sample use case of Olive to optimize a ResNet model using onnx conversion and Intel® Neural Compressor quantization tuner.

Performs optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Intel® Neural Compressor Quantized Onnx Model

Outputs the best metrics, model, and corresponding Olive config.

## Prerequisites
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample
### Run with both Intel® Neural Compressor static and dynamic quantization
run with config
```
python -m olive.workflows.run --config bert_inc_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_inc_config.json")
```

### Run with Intel® Neural Compressor static quantization
run with config
```
python -m olive.workflows.run --config bert_inc_static_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_inc_static_config.json")
```

### Run with Intel® Neural Compressor dynamic quantization
run with config
```
python -m olive.workflows.run --config bert_inc_dynamic_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_inc_dynamic_config.json")
```