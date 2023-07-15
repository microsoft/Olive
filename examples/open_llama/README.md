# Open LLaMa Optimization
This folder contains examples of Open LLaMA workflow.

This workflow also demonstrates how to use:
- Huggingface `transformers` to load model from [model hub](https://huggingface.co/models).
- Huggingface `optimum` to convert and merge generative models [optimum](https://huggingface.co/docs/optimum/index).

This example config file [open_llama_config.json](open_llama_config.json) is meant to be a starting point for optimizing Open LLaMA for target hardware. One can add additional passes as well as set different options for Transformer Optimization pass as per need. See [Olive documentation](https://microsoft.github.io/Olive/) for more information on optimizations passes available.

Note that this example config uses [openlm-research/open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b) for demonstration purpose. There are other models available in [Open LLaMA](https://huggingface.co/openlm-research) that can be used for optimization. The following table shows a few models' configurations:

| Model | Num Hidden Layers| Num Attention Heads | Hidden Size |
| --- | --- | --- | --- |
| [openlm-research/open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b) | 26 | 32 | 3200 |
| [openlm-research/open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b) | 32 | 32 | 4096 |
| [openlm-research/open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b) | 40 | 40 | 5120 |


When you run the example config for other larger models, you may need
1. change the `model_path` to the one you use.
```json
"input_model":{
    "type": "OptimumModel",
    "config": {
        "model_path": "openlm-research/open_llama_3b", // to change based on the model you use
        "model_components": ["decoder_model.onnx", "decoder_with_past_model.onnx"],
        "hf_config": {
            "model_class": "LlamaForCausalLM"
        }
    }
}
```
2. change the transformer optimization pass options in `open_llama_config.json` based on the above table:
```json
"optimize": {
    "type": "OrtTransformersOptimization",
    "config": {
        "model_type": "gpt2",
        "float16": true,
        "use_gpu": false,
        "keep_io_types": true,
        "num_heads": 32, // to change based on the model you use
        "hidden_size": 4096, // to change based on the model you use
        "optimization_options": {
            "use_multi_head_attention": false
        }
    }
}
```
3. increase the `num_hidden_layers` for dummy inputs in `user_script.py`.
```python
// to increase `num_hidden_layers` to conduct proper inputs data
def dummy_inputs(batch_size, torch_dtype, model_framework=Framework.PYTORCH, num_hidden_layers=26):
    past_sequence_length = 1
    attention_mask_sequence_length = 1
```

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
