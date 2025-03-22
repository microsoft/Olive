# Qwen 2.5 Optimization

Sample use cases of Olive to optimize a [Qwen 2.5 1.5B Instruct](Qwen/Qwen2.5-1.5B-Instruct) model using Olive.
- [Optimize for Qualcomm NPU](../phi3_5/README.md):
  - Replace `model_path` in `config.json` with `Qwen/Qwen2.5-1.5B-Instruct`.
  - Chat template is `"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"`
