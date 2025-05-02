# Vision Transformer (ViT) Optimization
This folder contains examples of ViT optimization using different workflows.
- Intel® NPU: [ViT base patch16 224 static shape model](#static-shape-model)

# ViT Base Patch16 224 Quantization

This folder contains a sample use case of Olive to optimize a [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) model using OpenVINO tools.

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Static shape model

The config file: [vit_base_patch16_224_context_ov_static.json](vit_base_patch16_224_context_ov_static.json) executes the above workflow producing static shape model.

## How to run

### Pip requirements

Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

**NOTE:**

- Access to the [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset is gated and therefore you will need to request access to the dataset. Once you have access to the dataset, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model

```bash
olive run --config vit_base_patch16_224_context_ov_static.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("vit_base_patch16_224_context_ov_static.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
