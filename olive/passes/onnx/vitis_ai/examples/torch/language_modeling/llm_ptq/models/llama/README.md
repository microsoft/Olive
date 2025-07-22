# Llama Models

## Benchmark

### Evaluation dataset

[wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext)

### Linux

**A100** GPU Mode, torch==2.2.0, transformers==4.37.2

| Model Name                | FP16  (Without Quantization)   | FP8  | FP8+FP8_KV_CACHE | W_UIN4(Per group)+A_BF16 | W_UIN4(Per group)+A_FP16+AWQ | W_UIN4(Per group)+A_FP16+GPTQ | W_UIN4(Per group)+A_FP16+SmoothQuant | W_INT8+A_INT8 |
| ------------------------- | ------------------------------ | ---- | ---------------- | ------------------------ | ---------------------------- | ----------------------------- | ------------------------------------ | ------------- |
|meta-llama/Llama-2-7b-hf   | 5.47                           | 5.54 | 5.51             |5.72                      | 5.61                         | 5.58                          | 5.68                                 | 19.09         |

**MI210** GPU Mode, torch==2.2.0+rocm5.7, transformers==4.37.2

| Model Name                | FP16 (Without Quantization) | FP8  | FP8+FP8_KV_CACHE | W_UIN4(Per group)+A_BF16 | W_UIN4(Per group)+A_FP16+AWQ | W_UIN4(Per group)+A_FP16+GPTQ | W_UIN4(Per group)+A_FP16+SmoothQuant | W_INT8+A_INT8 |
| ------------------------- | --------------------------- | ---- | ---------------- | ------------------------ | ---------------------------- | ----------------------------- | ------------------------------------ | ------------- |
| meta-llama/Llama-2-7b-hf  | 5.47                        | 5.54 | 5.51             | 5.72                     | 5.61                         | 5.58                          |5.96                                  | 18.52         |

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
