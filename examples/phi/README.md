# Phi optimization

- Phi/Phi-1.5 inference optimization with ONNX Runtime DirectML, go to [this example](https://github.com/microsoft/Olive/tree/main/examples/directml/phi)
- [Phi-1.5 Fine-tune optimization using QLoRA](#fine-tune-phi-15-model-on-a-code-generation-dataset-using-qlora)


### Fine-tune  phi-1.5 Model on a code generation dataset using QLoRA
This workflow fine-tunes [phi-1.5 model](https://huggingface.co/microsoft/phi-1_5) using [QLoRA](https://arxiv.org/abs/2305.14314) to generate code given a prompt.

The relevant config file is [phi_qlora_tinycodes.json](phi_qlora_tinycodes.json). The code language is set to `Python` but can be changed to other languages by changing the `language` field in the config file.
Supported languages are Python, TypeScript, JavaScript, Ruby, Julia, Rust, C++, Bash, Java, C#, and Go. Refer to the [dataset card](https://huggingface.co/datasets/nampdn-ai/tiny-codes) for more details on the dataset.

Note:
- You must be logged in to HuggingFace using `huggingface-cli login` to download the dataset or update `token` field in the config file with your HuggingFace token.
- This model doesn't support gradient_checkpointing yet.

Requirements file: [requirements-lora.txt](requirements-lora.txt)

## How to run
### Pip requirements
Install the necessary python packages using the corresponding requirements file.
```
python -m pip install -r <requirements_file>.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```
python -m olive.workflows.run --config <config_file>.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config <config_file>.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
