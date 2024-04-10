# ResNet optimization with PTQ on CPU
This is a sample use case of Olive to optimize a ResNet model using onnx conversion and onnx quantization tuner.

## Prerequisites
Please go to example repository [Quickstart ResNet Example](https://github.com/microsoft/Olive/tree/main/examples/resnet)

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Prepare data and model
To Prepare the model and necessary data:
```
python prepare_model_data.py --num_epochs 5
```

## Run sample using config
First, install required packages according to passes.
```
olive-cli run --config resnet_ptq_cpu.json --setup
```
Then, optimize the model
```
olive-cli run --config resnet_ptq_cpu.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("resnet_ptq_cpu.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
