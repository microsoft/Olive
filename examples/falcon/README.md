# Falcon Optimization
This folder contains a sample use case of Olive to optimize a [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) model using ONNXRuntime tools.

## Optimization Workflows
This workflow performs Falcon optimization on CPU with ONNX Runtime. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16*

Config file: [config.json](config.json)

## How to run
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```
olive run --config config.json --setup
```

Then, optimize the model
```
olive run --config config.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("config.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
