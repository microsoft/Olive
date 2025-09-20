# Deepseek R1 Distill optimization

Sample use cases of Olive to optimize a [DeepSeek R1 Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using Olive.

- [Finetune and Optimize for CPU/CUDA](../getting_started/olive-deepseek-finetune.ipynb)
- [QDQ Model with 4-bit Weights & 16-bit Activations](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config.json -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -o models/deepseek-r1-qdq`.
- [AMD NPU: Optimization and Quantization with for VitisAI](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config_vitis_ai.json -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -o models/deepseek-r1-vai`.
- [QUALCOMM NPU: PTQ + AOT Compilation using QNN EP](../phi3_5/README.md):
  - Run the workflow with `olive run --config qnn/DeepSeek_R1_Distill_Qwen_1.5B_qnn_config.json`.
  - Run the inference with `python app.py -m models/deepseek-r1-qnn -c "<｜User｜>{input}<｜Assistant｜><think>"`.
- [PTQ + AWQ ONNX OVIR Encapsulated 4-bit weight compression using Optimum OpenVINO](./openvino/)
