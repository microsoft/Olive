# CLIP VIT Optimization
This folder contains examples of CLIP VIT optimization using different workflows.

- NPU: [Optimization with PTQ on Qualcomm NPU using QNN EP](#clip-vit-optimization-with-ptq-on-npu)

Go to [How to run](#how-to-run)


## Optimization Workflows

### CLIP VIT optimization with PTQ on NPU
This workflow performs CLIP VIT optimization on Qualcomm NPU with ONNX Runtime PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Static shaped Onnx Model -> Quantized Onnx Model*

It requires x86 python environment on a Windows ARM machine with `onnxruntime-qnn` installed.

OpenAI clip model config file: [openai_clip-vit-base-patch16_ptq_qnn.json](openai_clip-vit-base-patch16_ptq_qnn.json)
 [openai_clip-vit-base-patch32_ptq_qnn.json](openai_clip-vit-base-patch32_ptq_qnn.json)
 [openai_clip-vit-large-patch14_ptq_qnn.json](openai_clip-vit-large-patch14_ptq_qnn.json)

Open clip model config file: [laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qnn.json](laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qnn.json)

**NOTE:** The model optimization part of the workflow can also be done on a Linux/Windows machine with a different onnxruntime package installed. Remove the `"evaluators"` and `"evaluator"` sections from the configuration file to skip the evaluation step.

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
