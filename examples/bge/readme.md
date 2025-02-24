# BAAI/bge-small-en-v1.5 Optimization

This folder contains examples of [BAAI/bge-small-en-v1.5 ](https://huggingface.co/BAAI/bge-small-en-v1.5) optimization using different workflows.

- NPU: [Optimization with PTQ using QNN EP](#ptq-using-qnn-ep)

## Optimization Workflows

### PTQ using QNN EP

This workflow performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Static shaped Onnx Model -> Quantized Onnx Model*

The precision will drop when Add or Softmax types of op are quantized, so they are not included.

| Quantized Ops | precision | latency (avg) |
|-|-|-|
| None (original model) | 0.8574675324675324 | N/A |
| All ("Mul", "Transpose", "Unsqueeze", "Add", "Softmax", "Gelu", "LayerNormalization", "Gather", "MatMul", "Sub", "Where", "Expand", "Gemm", "Tanh", "Reshape") | 0.19707792207792205 | 24.95298 |
| Without Softmax | 0.19675324675324674 | 24.08456 |
| Without Add | 0.1968831168831169 | 64.3278 |
| Without Add, Softmax | 0.8511038961038961 | 40.48591 |

TODO(anyone): debug Add and Softmax to add them back to improve latency

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
