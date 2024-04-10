# GPT-J Optimization Using Intel® Neural Compressor
This folder contains examples of [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) optimization using Intel® Neural Compressor.

This workflow demonstrates the capabilities of Intel® Neural Compressor on large language model.

## Optimization Workflows
This workflow performs GPT-J optimization on CPU with Intel® Neural Compressor PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Intel® Neural Compressor Quantized Onnx Model*

Config file: [gptj_inc_static_ptq_cpu.json](gptj_inc_static_ptq_cpu.json), [gptj_inc_dynamic_ptq_cpu.json](gptj_inc_dynamic_ptq_cpu.json)


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
olive-cli run --config <config_file>.json --setup
```

Then, optimize the model
```
olive-cli run --config <config_file>.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
