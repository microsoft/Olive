# BERT optimization with QAT Customized Training Loop on CPU
This folder contains a sample usecase of Olive using quantizaiton aware traning, Onnx conversion, ONNX Runtime transformers optimization,
ONNX Runtime performance tuning.

Performs optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> Transformers Optimized Onnx Model -> ONNX Runtime performance tuning*

Outputs a summary of the accuracy and latency metrics for each model.

## Prerequisites
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample using config
```
python -m olive.workflows.run --config bert_config.json
```
