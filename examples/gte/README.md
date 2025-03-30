# GTE-Large-v1.5 Optimization
This folder contains a sample use case of Olive to optimize a [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) model.

## Optimization Workflows
This workflow performs optimization on CPU with ONNX Runtime. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

Config file: [config.json](config.json)

## How to run
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
