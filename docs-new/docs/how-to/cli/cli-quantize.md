# Quantize

The `olive quantize` command enables you to try different quantization methods.

## :material-clock-fast: Quickstart

```bash
olive quantize \ 
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \ 
    --algorithms awq \ 
    --log_level 1
```