## Quantization of ResNet models through Brevitas using Quark interface

This is a minimal example on how to quantize an image classification model using Quark, while using the quantization infrastructure from the [Brevitas](https://github.com/Xilinx/brevitas) library.

Quark provides an interface to Brevitas, following an API close to the `quark.torch` quantization API.

In this example, we quantize a resnet model through Quark API for Brevitas quantizer with weight-only quantization and evaluate it on a subset of the ImageNet validation set, running:

```bash
python quantize_brevitas.py --model resnet50 --quant_scheme w_int8_per_tensor_sym --evaluation_samples 1000
```

For quick testing, evaluation can be done on a subset of the validation set using the argument `--evaluation_samples` (example: `--evaluation_samples 1000`).

For complete evaluation, the ImageNet validation needs to be downloaded first. With an [Hugging Face](https://huggingface.co) account and well configured read token (see the [login](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login) documentation), run:

```bash
huggingface-cli download ILSVRC/imagenet-1k data/val_images.tar.gz --repo-type dataset --local-dir imagenet
cd imagenet
tar -xf val_images.tar.gz -C validation
```

and the argument `--data_dir /path/to/imagenet` can then be used.

To run this example with activation quantization (both `nn.Linear` inputs and outputs), use the following:

```bash
python quantize_brevitas.py --model resnet50 --quant_scheme w_int8_a_int8_per_tensor_sym --data_dir /path/to/imagenet
```

## Results

The results of the evaluation on the full ImageNet-1k validation dataset (50k images) are reported below for [torchvision resnet50 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) (`IMAGENET1K_V1` checkpoint).

| quant_scheme                 | Accuracy |
|------------------------------|----------|
| None (float32 checkpoint)    | 76.144   |
| w_int8_per_tensor_sym        | 75.836   |
| w_int8_a_int8_per_tensor_sym | 75.804   |

* `w_int8_per_tensor_sym`: Weight-only quantization, symmetric, per tensor, narrow range.
* `w_int8_a_int8_per_tensor_sym`: on top of the above weight quantization, the inputs of [supported layers](/quark/torch/extensions/brevitas/api.py#L190) are quantized, with per tensor symmetric quantization, no narrow range.

Note that in these evaluations we use torchvision checkpoints for resnet rather than Transformers' [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) which proved to be more sensitive to quantization.
