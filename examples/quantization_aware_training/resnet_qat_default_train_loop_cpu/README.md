# ResNet optimization with QAT Default Training Loop on CPU
This folder contains a sample use case of Olive using quantization aware training, onnx conversion, ONNX Runtime performance tuning.

Performs optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> ONNX Runtime performance tuning*

Outputs a summary of the accuracy and latency metrics for each model.

## Prerequisites
### Prepare data and model
```
python prepare_model_data.py
```
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample using config
First, install required packages according to passes.
```
python -m olive.workflows.run --config resnet_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config resnet_config.json
```
