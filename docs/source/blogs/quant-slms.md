# Finding the Best Settings for Quantized Models Using Olive

The advent of small language models (SLMs) has enabled the deployment of AI applications on edge devices, such as smartphones and IoT devices. However, these models often require optimization to run efficiently on resource-constrained hardware. Quantization is a popular technique for reducing the size and computational requirements of neural networks by converting weights and activations from floating-point to lower-precision formats, such as int4 or int8.

Olive has multiple features that allow users to easily experiment with different quantization settings and find the best configuration for their specific use case. Users can quantize the weights easily on the source PyTorch models using techniques such as [GPTQ](https://arxiv.org/abs/2210.17323) that take advantage of the model's structure to minimize the loss in accuracy. Olive can also quantize the weights and activations of the model using post-training quantization (PTQ) on the ONNX model. It also has built-in support for the popular [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmark suite, which allows users to evaluate the performance of their quantized models on a variety of tasks efficiently.

In this experiment we use Olive to search for the best settings from some key models of interest along the following dimensions for weight-only quantization:
- To perform weight rotation using [QuaRot](https://arxiv.org/abs/2404.00456) or not
- Mixed precision quantization (e.g., int4 for some layers and int8 for others)
- GPTQ settings:
    - symmetric vs asymmetric quantization
    - group size: 32, 128

The dimension of most interest to us for this experiment is the mixed precision setting. Full int4 quantization can lead to significant accuracy degradation for some models, while mixed precision quantization can help to mitigate this issue by allowing users to choose the precision for each layer. This however comes at the cost of increased size and inference time. Olive currently uses a simple heuristic to determine which layers to keep in higher precision (int8) and provides the following options:
- No mixed precision (all layers in int4)
- k_quant_down: lm head in int8
- k_quant_down: lm head + mlp down projection for first 1/8, last 1/8 and every thrird middle layer in int8
- k_quant_mixed: lm head + mlp down and attention qkv projections for first 1/8, last 1/8 and every thrird middle layer in int8

## Experiment Setup
We ran Olive sweeps over the quantization settings above using an end-to-end workflow that includes the following steps:
1. Optional QuaRot Pass
2. GPTQ
3. Export model to ONNX
4. Evaluate using lm-evaluation-harness (optionally on GPU using IO-Binding for faster evaluation)

We evaluated the models on various tasks from the benchmark suite - ARC-Easy, ARC-Challenge, MMLU, MMLU-STEM, HellaSwag, and OpenBookQA. We compared the models against the original float16 models and the baseline quantized models available through [FoundryLocal](https://github.com/microsoft/Foundry-Local).

## Results
We ran the experiment on the following models:
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [Llama‑3.2‑1B‑Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Phi‑3.5‑mini‑instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [Phi‑4‑mini‑instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Qwen2.5‑1.5B‑Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

The following table summarizes the best settings found for each model along with comparisons with the original and baseline models.
| Base model | Best schema | Δ mean quality vs Original (%) | Δ mean quality vs Baseline (%) | Size (GB) | Size vs Original (%) | Size vs Baseline (%) |
|---|---|---:|---:|---:|---:|---|
| deepseek‑ai/DeepSeek‑R1‑Distill‑Qwen‑1.5B | block=128, symmetric=False, smp=k_quant_mixed, quarot=False |  +2.724 | +7.624 | 1.434 | -57.087 | +0.005 |
| meta‑llama/Llama‑3.2‑1B‑Instruct | block=32, symmetric=False, smp=k_quant_down, quarot=True | -0.096 | +7.342 | 1.361 | -51.505 | +18.112 |
| microsoft/Phi‑3.5‑mini‑instruct | block=32, symmetric=False, smp=k_quant_down, quarot=False | +0.924 | +1.042 | 2.453 | -65.650 | +13.666 |
| microsoft/Phi‑4‑mini‑instruct | block=32, symmetric=False, smp=k_quant_mixed, quarot=True |  -0.348 | +0.929 | 3.884 | -53.762 | +6.169 |
| Qwen/Qwen2.5‑1.5B‑Instruct | block=32, symmetric=True, smp=k_quant_down, quarot=False | +0.547 | +1.636 | 1.479 | -55.432 | +18.155 |

The results show that Olive was able to find quantization settings that significantly improve the performance of the quantized models compared to the baseline models. In some cases, the quantized models even outperformed the original float16 models.

Some key observations from the results:
- Block size: A smaller block size (32) generally led to better performance compared to a larger block size (128). Likely due to the increased granularity in quantization.
- Mixed precision: `k_quant_down` was the most consistently strong (Llama‑1B, Phi‑3.5‑mini, Qwen2.5‑1.5B). It is a good trade-off between size and quality.
- Symmetric vs Asymmetric: Asymmetric quantization (symmetric=False) was preferred for most models. Asymmetric quantization can better capture the distribution of weights since it can use the full range of the quantized data type.
- QuaRot: The QuaRot pass was beneficial for some models (Llama‑1B, Phi‑4‑mini) but not for others. This suggests that the effectiveness of QuaRot is model-dependent.

## Why Olive?
Olive provides a flexible and extensible framework for experimenting with different quantization settings. Its integration with the ONNX Runtime allows for efficient evaluation of quantized models, making it easy to iterate and find the best configuration. The built-in support for popular benchmarks like lm-evaluation-harness further simplifies the evaluation process.

Each setting is a knob in the Olive configuration file, making it easy to set up and run experiments. Olive also provides a built-in hyperparameter sweep capability, allowing users to explore a wide range of configurations systematically.
