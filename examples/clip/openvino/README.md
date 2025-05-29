# CLIP-ViT-B-32-laion2B-s34B-b79K Quantization

This folder contains a sample use case of Olive to optimize a [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K), [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) and [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) models using OpenVINO tools.

- Intel® NPU: [CLIP-ViT-B-32-laion2B-s34B-b79K static shape model](#clip-vit-b-32-laion2b-s34b-b79k-static-shape-model)
- Intel® NPU: [CLIP ViT Base patch16 static shape model](#clip-vit-base-patch16-static-shape-model)
- Intel® NPU: [CLIP ViT Base patch32 static shape model](#clip-vit-base-patch32-static-shape-model)

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### CLIP-ViT-B-32-laion2B-s34B-b79K Static shape model

The config file: [clip_vit_b32_laion2b_s34B_b79k_context_ov_static.json](clip_vit_b32_laion2b_s34B_b79k_context_ov_static.json) executes the above workflow producing static shape model.

## CLIP ViT Base patch16 static shape model

The config file: [clip_vit_base_patch16_context_ov_static.json](clip_vit_base_patch16_context_ov_static.json) executes the above workflow producing static shape model.

## CLIP ViT Base patch32 static shape model

The config file: [clip_vit_base_patch32_context_ov_static.json](clip_vit_base_patch32_context_ov_static.json) executes the above workflow producing static shape model.

## How to run

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

```bash
olive run --config <config_file>.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
