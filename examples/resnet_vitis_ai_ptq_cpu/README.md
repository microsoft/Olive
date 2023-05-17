# ResNet optimization with PTQ on CPU
This folder contains a sample use case of Olive to optimize a ResNet model using onnx conversion and vitis ai onnx quantization tools.

## Prerequisites
### Prepare data and model
To Prepare the model and necessary data:
```
python prepare_model_data.py --num_epochs 5
```

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample using config
First, install required packages according to passes.
```
python -m olive.workflows.run --config resnet_static_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config resnet_static_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("resnet_static_config.json")
```
