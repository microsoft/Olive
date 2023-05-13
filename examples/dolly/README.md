# Dolly Optimization
This folder contains a sample use case of Olive to optimize a [dolly](https://huggingface.co/databricks/dolly-v2-12b) model using onnx conversion.

Performs optimization pipeline:
- *PyTorch Model -> Onnx Model

## Prerequisites
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample using config. The optimization techniques to run are specified in dolly_config.json
First, install required packages according to passes.
```
python -m olive.workflows.run --config dolly_config.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config dolly_config.json
```
