# Open LLaMa Optimization
This folder contains examples of Open LLaMA workflow.

## Optimization Workflows
### Convert, Optimize and Merge Open LLaMA Model
This workflow also demonstrates how to use:
- Huggingface `transformers` to load model from [model hub](https://huggingface.co/models).
- Huggingface `optimum` to convert and merge generative models [optimum](https://huggingface.co/docs/optimum/index).
- Intel® Neural Compressor `neural-compressor` to compress model with 4-bits weight-only quantization ([WOQ](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md)).

This example config file [open_llama_config.json](open_llama_config.json) is meant to be a starting point for optimizing Open LLaMA for target hardware. One can add additional passes as well as set different options for Transformer Optimization pass as per need. See [Olive documentation](https://microsoft.github.io/Olive/) for more information on optimizations passes available.

Note that this example config uses [openlm-research/open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b) for demonstration purpose. There are other models available in [Open LLaMA](https://huggingface.co/openlm-research) that can be used for optimization. The following table shows a few models' configurations:

| Model | Num Hidden Layers| Num Attention Heads | Hidden Size |
| --- | --- | --- | --- |
| [openlm-research/open_llama_3b](https://huggingface.co/openlm-research/open_llama_3b) | 26 | 32 | 3200 |
| [openlm-research/open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b) | 32 | 32 | 4096 |
| [openlm-research/open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b) | 40 | 40 | 5120 |

Requirements file: [requirements.txt](requirements.txt)

When you run the example config for other larger models, you may need
1. change the `model_path` to the one you use in `open_llama_config.json` and `user_script.py`.
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
    ```python
    import torch
    from transformers import AutoConfig

    from olive.constants import Framework

    model_id = "openlm-research/open_llama_3b" # to change based on the model you use
    config = AutoConfig.from_pretrained(model_id)
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

### Sparsify Open LLaMA Model using SparseGPT
This workflow sparsifies Open LLaMA model using [SparseGPT](https://arxiv.org/abs/2301.00774). The output model is still a transformers pytorch model but with the layer weights
sparsified. The given config has sparsity set to `[2,4]` for [structured 2:4 sparsity pattern](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) but
can be changed to other sparsity pattern such as `0.5` for 50% unstructured sparsity or `[4,8]` for 4:8 structured sparsity pattern.

To take advantage of the sparsity using TensorRT, the sparse `torch.nn.Linear` modules in the transformer layers are then converted to `TRTModule` from `torch-tensorrt` with fp16 precision and sparsity enabled.
This is done using the `TorchTRTConversion` pass in Olive which saves the entire model. This saved model can then be loaded using `torch.load` but requires Olive to be installed.
Inference is done like a normal pytorch model.

The relevant config file is [open_llama_sparsegpt_gpu.json](open_llama_sparsegpt_gpu.json)

Requirements file: [requirements-sparsegpt.txt](requirements-sparsegpt.txt)

### Fine-tune Llama Model on a chatbot dataset using QLoRA
This workflow fine-tunes LLaMA model using [QLoRA](https://arxiv.org/abs/2305.14314). The output model is still the input transformers model along with a quantization config and
LoRA adapters that were fine-tuned on the training dataset.

The relevant config file is [llama_qlora.json](llama_qlora.json). It corresponds to the [guqnaco 7b example in the original qlora implementation](https://github.com/artidoro/qlora/blob/main/scripts/finetune_guanaco_7b.sh).

Requirements file: [requirements-qlora.txt](requirements-qlora.txt)

### Fine-tune Open Llama Model on a code generation dataset using QLoRA
This workflow fine-tunes Open LLaMA model using [QLoRA] to generate code given a prompt.

The relevant config file is [open_llama_qlora_tinycodes.json](open_llama_qlora_tinycodes.json). The code language is set to `Python` but can be changed to other languages by changing the `language` field in the config file.
Supported languages are Python, TypeScript, JavaScript, Ruby, Julia, Rust, C++, Bash, Java, C#, and Go. Refer to the [dataset card](https://huggingface.co/datasets/nampdn-ai/tiny-codes) for more details on the dataset.

Note: You must be logged in to HuggingFace using `huggingface-cli login` to download the dataset or update `token` field in the config file with your HuggingFace token.

Requirements file: [requirements-qlora.txt](requirements-qlora.txt)

### Optimizing Open Llama Model with Azure Arc
This workflow optimizes Open Llama model on Azure ML compute, and evaluate output models on your device. Please connect your device to Azure Arc by following instruction: [Self-hosted Kubernetes cluster](https://microsoft.github.io/Olive/tutorials/azure_arc.html)

This example config file is [open_llama_arc.json](open_llama_arc.json).

Requirements file: [requirements-arc.txt](requirements-arc.txt)

### Compress Open Llama Model with Intel® Neural Compressor 4-bits Weight-only Quantization
This workflow compresses Open Llama model with 4-bits weight-only quantization using Intel® Neural Compressor, and evaluate accuracy and perplexity on [lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai) datasets.

This example config file is [open_llama_inc_woq.json](open_llama_inc_woq.json).

Requirements file: [requirements-woq.txt](requirements-arc.txt)

#### Prerequisites

To use Intel® Neural Compressor 4-bits weight-only quantization, please install `neural-compressor>=2.3`. Weight-only quantization in Intel® Neural Compressor is still under development. We encourage you to use the master branch to access the latest features. Please check the link of [installing neural-compressor from source](https://github.com/intel/neural-compressor/blob/master/docs/source/installation_guide.md#install-from-source-1)

#### Run 4-bits weight-only quantization

4-bits weight-only quantization supports two algorithms:
- Round-to-nearest (RTN) is the most straightforward way to quantize weight using scale maps.
- GPTQ algorithm provides more accurate quantization but requires more computational resources.

To compress model with 4-bits weight-only quantization, you may need
1. set `approach` to `weight_only`, and set `algorithm` to `GPTQ` or `RTN` in `weight_only_config`.
```json
"quantization": {
    "type": "IncStaticQuantization",
    "config": {
        "user_script": "user_script.py",
        "approach": "weight_only",
        "weight_only_config":{
            "algorithm": "RTN"
        }
    }
}
```
2. if `GPTQ` algorithm is used, you need to provide a calibration dataloader, which outputs input data and label.
```json
"quantization": {
    "type": "IncStaticQuantization",
    "config": {
        "user_script": "user_script.py",
        "approach": "weight_only",
        "weight_only_config":{
            "algorithm": "GPTQ"
        }
    },
    "dataloader_func": "calib_dataloader",
}
```
```python
class CalibDataloader:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.dataset = []
        # operations to add (input_data, label) pairs into self.dataset

    def __iter__(self):
        for input_data, label in self.dataset:
            yield input_data, label
```

#### Validated results

The following table shows the accuracy and perplexity results of Open Llama models evaluated on lambada_openai task. `GPTQ W4G32Asym` in the configuration column means GPTQ algorithm is used for 4-bits weight only quantization, setting group_size=32 and scheme=asym.

<table class="tg">
<thead>
  <tr>
    <th rowspan="2">Model name</th>
    <th rowspan="2">Configuration</th>
    <th colspan="2">Lambada_openai</th>
    <th rowspan="2">Accuracy Ratio<br>[WOQ/FP32]</th>
  </tr>
  <tr>
    <th>Accuracy</th>
    <th>Perplexity</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">openlm-research/open_llama_3b</td>
    <td>FP32</td>
    <td>0.6637</td>
    <td>4.8496</td>
    <td>/</td>
  </tr>
  <tr>
    <td>GPTQ<br>W4G32Asym</td>
    <td>0.6579</td>
    <td>4.9773</td>
    <td>99.13%</td>
  </tr>
  <tr>
    <td rowspan="2">openlm-research/open_llama_7b</td>
    <td>FP32</td>
    <td>0.7044</td>
    <td>3.9716</td>
    <td>/</td>
  </tr>
  <tr>
    <td>GPTQ<br>W4G32Sym</td>
    <td>0.7017</td>
    <td>4.1320</td>
    <td>99.62%</td>
  </tr>
  <tr>
    <td rowspan="2">openlm-research/open_llama_13b</td>
    <td>FP32</td>
    <td>0.7213</td>
    <td>3.5728</td>
    <td>/</td>
  </tr>
  <tr>
    <td>RTN<br>W4G32Sym</td>
    <td>0.7174</td>
    <td>3.7322</td>
    <td>99.36%</td>
  </tr>
</tbody>
</table>

> Note: The above results are obtained using onnxruntime built from source code with the `sub_byte_quant_zp` branch, which enables support for the `MatMulWithQuantWeight` op. Weight-only quantization in Intel® Neural Compressor is still under development. We encourage you to use the master branch to access the latest features.


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
