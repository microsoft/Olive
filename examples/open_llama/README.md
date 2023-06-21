# BERT Optimization
This folder contains examples of Open LLaMA workflow.

This workflow also demonstrates how to use:
- Huggingface `transformers` to load model from [model hub](https://huggingface.co/models).
- Huggingface `optimum` to convert and merge generative models [optimum](https://huggingface.co/docs/optimum/index).

This example config file [open_llama_config.json](open_llama_config.json) is meant to be a starting point for optimizing Open LLaMA for target hardware. One can add additional passes as well as set different options for Transformer Optimization pass as per need. See [Olive documentation](https://microsoft.github.io/Olive/) for more information on optimizations passes available.

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```
python -m olive.workflows.run --config open_llama_config.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config open_llama_config.json
```


or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("open_llama_config.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
