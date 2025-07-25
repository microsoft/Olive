# OPT Models

## Benchmark

### Evaluation dataset

[wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext)

### Linux

**A100** GPU Mode, torch==2.2.0, transformers==4.37.2

| Model Name                | FP16 (Without Quantization)  | FP8   | FP8+FP8_KV_CACHE | W_UIN4(Per group)+A_BF16 | W_UIN4(Per group)+A_FP16+AWQ | W_UIN4(Per group)+A_FP16+GPTQ | W_UIN4(Per group)+A_FP16+SmoothQuant | W_INT8+A_INT8 |
| ------------------------- | ---------------------------- | ----  | ---------------- | ------------------------ | ---------------------------- | ----------------------------- | ------------------------------------ | ------------- |
| facebook/opt-125m         | 27.65                        | 28.11 | 28.0             | 30.49                    | 29.77                        | 29.39                         | 31.26                                | 30.18         |

**MI210** GPU Mode, torch==2.2.0+rocm5.7, transformers==4.37.2

| Model Name                | FP16 (Without Quantization) | FP8   | FP8+FP8_KV_CACHE | W_UIN4(Per group)+A_BF16 | W_UIN4(Per group)+A_FP16+AWQ | W_UIN4(Per group)+A_FP16+GPTQ | W_UIN4(Per group)+A_FP16+SmoothQuant | W_INT8+A_INT8 |
| ------------------------- | --------------------------- | ----- | ---------------- | ------------------------ | ---------------------------- | ----------------------------- | ------------------------------------ | ------------- |
| facebook/opt-125m         | 27.64                       | 28.08 | 27.99            | 30.46                    | 29.74                        | 29.29                         | 31.24                                | 30.21         |
| facebook/opt-13b          | 10.12                       | 10.24 | 10.24            | 10.3                     | 10.43                        | 10.18                         | 10.39                                | 4135.99       |

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
