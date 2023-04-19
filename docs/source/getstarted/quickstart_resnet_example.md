# ResNet optimization with PTQ on CPU
This is a sample use case of Olive to optimize a ResNet model using onnx conversion and onnx dynamic/static quantization tuner.

## Prerequisites
Please go to example repository [Quickstart ResNet Example](https://github.com/microsoft/Olive/tree/main/examples/resnet_ptq_cpu)
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
python -m olive.workflows.run --config resnet_{dynamic,static}_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config resnet_{dynamic,static}_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("resnet_dynamic_config.json")
olive_run("resnet_static_config.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
