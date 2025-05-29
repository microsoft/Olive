# Deepseek R1 Distill optimization

Sample use cases of Olive to optimize a [DeepSeek R1 Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using Olive.

- [Finetune and Optimize for CPU/CUDA](../getting_started/olive-deepseek-finetune.ipynb)
- [QDQ Model with 4-bit Weights & 16-bit Activations](../phi3_5/README.md):
  - Replace `model_path` in `qdq_config.json` with `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- [AMD NPU: Optimization and Quantization with for VitisAI](../phi3_5/README.md):
  - Replace `model_path` in `qdq_config_vitis_ai.json` with `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- [PTQ + AOT Compilation for Qualcomm NPUs using QNN EP](../phi3_5/README.md):
  - Replace `model_path` in `qnn_config.json` with `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
  - Chat template is `"<｜User｜>{input}<｜Assistant｜><think>"`.
- [PTQ + AWQ ONNX OVIR Encapsulated 4-bit weight compression using Optimum OpenVINO](./openvino/)
