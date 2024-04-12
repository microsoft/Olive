# AST Optimization
This folder contains examples of AST(Audio Spectrogram Transformer) optimization using olive workflows.

- CPU: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

- Model: https://huggingface.co/MIT/ast-finetuned-speech-commands-v2
- Dataset: https://huggingface.co/datasets/speech_commands

### Run example using config

The `ast.json` is used on CPU optimization which tries to quantize the model and tune the inference config for better performance.

First, install required packages according to passes.
```sh
olive run --config ast.json --setup
```

Then, optimize the model
```sh
olive run --config ast.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("ast.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
