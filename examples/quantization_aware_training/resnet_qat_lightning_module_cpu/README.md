# ResNet optimization with QAT PyTorch Lightning Module on CPU
This folder contains a sample usecase of Olive using quantizaiton aware traning, onnx conversion, ONNX Runtime performance tuning.

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
```
python -m olive.workflows.run --config resnet_config.json
```
