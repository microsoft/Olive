# Exploring Optimal Quantization Settings for Small Language Models with Olive

The rapid advancement of small language models (SLMs) has made it possible to deploy AI applications directly on edge devices such as smartphones and IoT hardware. However, running these models efficiently under tight memory and compute constraints requires careful optimization.

Quantization is one of the most effective optimization techniques for reducing model size and computation by converting weights and activations from floating-point to lower-precision formats such as INT8 or INT4. While quantization can drastically shrink memory and latency footprints, it often comes with trade-offs in accuracy and stability.

Olive provides a unified and extensible framework to explore these trade-offs efficiently. It enables users to apply and evaluate multiple quantization strategies, including [GPTQ](https://arxiv.org/abs/2210.17323) and post-training quantization (PTQ), and to benchmark results against standardized evaluation suites such as [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## Experiment Overview

In this experiment, we used Olive to find the best weight-only quantization settings across several key model families. The sweep focused on the following dimensions:

* **QuaRot**: whether to apply weight rotation using [QuaRot](https://arxiv.org/abs/2404.00456)
* **Mixed precision**: selectively using higher precision (e.g., INT8) for certain layers
* **GPTQ configuration:**

  * symmetric vs asymmetric quantization
  * group size (block size): 32 or 128

Among these, **mixed precision** was of particular interest. Full INT4 quantization can lead to large accuracy drops, while selectively keeping sensitive layers (e.g., LM head, MLP down-projection) in INT8 often helps preserve quality—with moderate increases in size and inference time.

Olive provides simple configuration options for these strategies:

* **No mixed precision:** all layers in INT4
* **`k_quant_down`:** LM head in INT8
* **`k_quant_down` (extended):** LM head + MLP down-projection for first 1/8, last 1/8, and every third middle layer in INT8
* **`k_quant_mixed`:** LM head + MLP down and attention QKV projections for first 1/8, last 1/8, and every third middle layer in INT8

---

## Experiment Setup

We used Olive to evaluate combinations of the above settings through the following pipeline:

1. (Optional) QuaRot weight rotation
2. GPTQ weight quantization
3. Model export to ONNX
4. Evaluation via `lm-evaluation-harness` (optionally accelerated using ONNX Runtime IO-Binding on GPU)

The evaluation covered a range of reasoning and commonsense tasks, including **ARC-Easy**, **ARC-Challenge**, **MMLU**, **MMLU-STEM**, **HellaSwag**, and **OpenBookQA**.
Results were compared against:

* The **original float16 models**, and
* The **baseline quantized models** available through [FoundryLocal](https://github.com/microsoft/Foundry-Local).

---

## Recipes

We ran the sweeps on the following SLMs using these recipes:

* [DeepSeek-R1-Distill-Qwen-1.5B](https://github.com/microsoft/olive-recipes/blob/main/deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/olive/README.md)
* [Llama-3.2-1B-Instruct](https://github.com/microsoft/olive-recipes/blob/main/meta-llama-Llama-3.2-1B-Instruct/olive/README-mixed.md)
* [Phi-3.5-mini-instruct](https://github.com/microsoft/olive-recipes/blob/main/microsoft-Phi-3.5-mini-instruct/olive/README.md)
* [Phi-4-mini-instruct](https://github.com/microsoft/olive-recipes/blob/main/microsoft-Phi-4-mini-instruct/olive/README.md)
* [Qwen2.5-1.5B-Instruct](https://github.com/microsoft/olive-recipes/tree/main/Qwen-Qwen2.5-1.5B-Instruct/olive)

---

## Results

| Base model                                | Best schema                                                 | Δ mean quality vs Original (%) | Δ mean quality vs Baseline (%) | Size (GB) | Size vs Original (%) | Size vs Baseline (%) |
| ----------------------------------------- | ----------------------------------------------------------- | -----------------------------: | -----------------------------: | --------: | -------------------: | -------------------: |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | block=128, symmetric=False, smp=k_quant_mixed, quarot=False |                         +2.724 |                         +7.624 |     1.434 |              −57.087 |               +0.005 |
| meta-llama/Llama-3.2-1B-Instruct          | block=32, symmetric=False, smp=k_quant_down, quarot=True    |                         −0.096 |                         +7.342 |     1.361 |              −51.505 |              +18.112 |
| microsoft/Phi-3.5-mini-instruct           | block=32, symmetric=False, smp=k_quant_down, quarot=False   |                         +0.924 |                         +1.042 |     2.453 |              −65.650 |              +13.666 |
| microsoft/Phi-4-mini-instruct             | block=32, symmetric=False, smp=k_quant_mixed, quarot=True   |                         −0.348 |                         +0.929 |     3.884 |              −53.762 |               +6.169 |
| Qwen/Qwen2.5-1.5B-Instruct                | block=32, symmetric=True, smp=k_quant_down, quarot=False    |                         +0.547 |                         +1.636 |     1.479 |              −55.432 |              +18.155 |

---

### Key Takeaways

* **Block size:** Smaller block sizes (32) consistently yielded better results than larger ones (128), likely due to finer granularity during quantization.
* **Mixed precision:** The `k_quant_down` configuration provided the best trade-off between quality and model size, performing strongly across most models.
* **Symmetric vs asymmetric:** Asymmetric quantization (`symmetric=False`) performed better for most models, as it captures asymmetric weight distributions more effectively.
* **QuaRot:** The QuaRot pass benefited certain models (e.g., Llama-1B, Phi-4-mini) but not others, indicating model-specific sensitivity.

Overall, Olive discovered quantization settings that significantly improved over baseline quantized models, and in some cases even surpassed the original float16 versions in task accuracy.

---

## Why Olive?

Olive simplifies and accelerates experimentation with quantization and deployment optimizations.
Its modular pipeline design and integration with ONNX Runtime allow users to efficiently test, benchmark, and deploy optimized models across hardware backends.

Each quantization dimension, such as block size, symmetry, mixed precision, or QuaRot, is exposed as a configurable parameter in Olive’s JSON configuration, enabling structured exploration and reproducibility. Olive’s built-in sweep capability helps users systematically explore a range of quantization settings and identify configurations that achieve a strong balance between model size, latency, and accuracy.

---

## Summary and What’s Next

This experiment showed that Olive can be used to explore and compare different quantization strategies efficiently. Across several small language models, we observed that mixed precision configurations can preserve most of the model quality while significantly reducing size and compute requirements. The choice of block size and symmetry type also plays an important role in overall performance.

Next, we are extending Olive’s quantization capabilities to include smarter mixed precision strategies. These will automatically analyze layer sensitivity to determine which layers benefit most from higher precision. This approach will reduce the need for manual configuration while maintaining accuracy and efficiency across a wider range of models.
