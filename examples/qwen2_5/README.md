# Qwen 2.5 Optimization

Sample use cases of Olive to optimize a [Qwen/Qwen 2.5 1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model using Olive.

- [QDQ Model with 4-bit Weights & 16-bit Activations](../phi3_5/README.md):
  - Replace `model_path` in `qdq_config.json` with `Qwen/Qwen2.5-1.5B-Instruct`.
- [AMD NPU: Optimization and Quantization with for VitisAI](../phi3_5/README.md):
  - Replace `model_path` in `qdq_config_vitis_ai.json` with `Qwen/Qwen2.5-1.5B-Instruct`.
- [PTQ + AOT Compilation for Qualcomm NPUs using QNN EP](../phi3_5/README.md):
  - Replace `model_path` in `qnn_config.json` with `Qwen/Qwen2.5-1.5B-Instruct`.
  - Chat template is `"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"`
- [PTQ + AWQ ONNX OVIR Encapsulated 4-bit weight compression using Intel® Optimum OpenVINO](./openvino/)
