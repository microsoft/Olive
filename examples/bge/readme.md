# BAAI/bge-small-en-v1.5 Optimization

This folder contains examples of [BAAI/bge-small-en-v1.5 ](https://huggingface.co/BAAI/bge-small-en-v1.5) optimization using different workflows.

- NPU: [Optimization with PTQ using QNN EP](#ptq-using-qnn-ep)

## Optimization Workflows

### PTQ using QNN EP

This workflow performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Static shaped Onnx Model -> Quantized Onnx Model*

| Model | precision | latency (avg) |
|-|-|-|
| Original model | 0.8574675324675324 | N/A |
| Quantized model | 0.8504870129870131 | 16.36546 |

## How to run
### Pip requirements
Install the necessary python packages:
```sh
# [NPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[qnn]
```

### Other dependencies
```sh
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```sh
olive run --config <config_file>.json --setup
```

Then, optimize the model
```sh
olive run --config <config_file>.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
