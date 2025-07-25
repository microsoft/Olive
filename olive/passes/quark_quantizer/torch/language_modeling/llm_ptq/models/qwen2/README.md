# Qwen1.5 Models

## Benchmark

### Evaluation dataset

[wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext)

### Linux

**A100** GPU Mode, torch==2.2.0, transformers==4.37.2

| Model Name                | BFloat16 (Without Quantization) | FP8   | FP8+FP8_KV_CACHE | W_UIN4(Per group)+A_BF16 | W_UIN4(Per group)+A_FP16+AWQ | W_UIN4(Per group)+A_FP16+GPTQ | W_UIN4(Per group)+~~A_FP16~~+SmoothQuant | W_INT8+A_INT8 |
| ------------------------- | ------------------------------- | ----- | ---------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------------------- | ------------- |
| Qwen_Qwen1.5-0.5B         | 14.8                            | 15.57 | 15.35            | 17.39                    | 16.39                        | 15.71                         |17.38                                     | 40.29         |
| Qwen_Qwen1.5-7B           | 7.95                            | 8.06  | 8.01             | 8.98                     | 8.22                         | 8.13                          |8.98                                      | 21.37         |

**MI210** GPU Mode, torch==2.2.0+rocm5.7, transformers==4.37.2

| Model Name                | BFloat16 (Without Quantization)  | FP8   | FP8+FP8_KV_CACHE | W_UIN4(Per group)+A_BF16 | W_UIN4(Per group)+A_FP16+AWQ | W_UIN4(Per group)+A_FP16+GPTQ | W_UIN4(Per group)+A_FP16+SmoothQuant | W_INT8+A_INT8 |
| ------------------------- | -------------------------------- | ----- | ---------------- | ------------------------ | ---------------------------- | ----------------------------- | ------------------------------------ | ------------- |
| Qwen_Qwen1.5-0.5B         | 14.8                             | 15.55 | 15.35            |17.39                     | 16.38                        | 15.71                         |17.38                                 | 40.1          |
| Qwen_Qwen1.5-7B           | 7.95                             | 8.06  | 8.01             |8.98                      | 8.22                         | 8.12                          |8.98                                  | 21.32         |

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
