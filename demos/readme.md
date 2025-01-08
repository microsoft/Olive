# Llama 1B sample

https://github.com/microsoft/Olive/blob/main/examples/getting_started/olive-awq-ft-llama.ipynb

- Use python 3.11 because autoawq 0.2.6 does not support general py3 https://pypi.org/project/autoawq/0.2.6/#files
- Install pytorch + cuda: https://pytorch.org/get-started/locally/
    + pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
- Setup `pip install -r demos\requirements.txt`
- Run `olive quantize --model_name_or_path meta-llama/Llama-3.2-1B-Instruct --trust_remote_code --algorithm awq --output_path models/llama/awq --log_level 1`
- Run `olive auto-opt --model_name_or_path models/llama/awq --device cpu --provider CPUExecutionProvider --use_ort_genai --output_path models/llama/onnx --log_level 1`

# Use auto-opt

```
olive auto-opt `
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct `
    --trust_remote_code `
    --output_path models/llama/ao `
    --device cpu `
    --provider CPUExecutionProvider `
    --use_ort_genai `
    --precision int4 `
    --log_level 1
```

# Others

Install triton: https://huggingface.co/madbuda/triton-windows-builds

QNN help: https://microsoft.github.io/Olive/how-to/configure-workflows/model-opt-and-transform/qnn.html
