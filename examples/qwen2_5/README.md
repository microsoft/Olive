# Qwen 2.5 Optimization

Sample use cases of Olive to optimize a [Qwen/Qwen 2.5 1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model using Olive.

- [QDQ Model with 4-bit Weights & 16-bit Activations](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config.json -m Qwen/Qwen2.5-1.5B-Instruct -o models/qwen2_5-qdq`.
- [AMD NPU: Optimization and Quantization with for VitisAI](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config_vitis_ai.json -m Qwen/Qwen2.5-1.5B-Instruct -o models/qwen2_5-vai`.
- [QUALCOMM NPU: Optimization and Quantization using QNN EP](../phi3_5/README.md):
  - Refer to the Qualcomm NPU section below.
- [PTQ + AWQ ONNX OVIR Encapsulated 4-bit weight compression using Intel® Optimum OpenVINO](./openvino/)

## **Optimization and Quantization for AMD NPU**

- [**AMD NPU**](./vitisai/): Instructions to run quantization and optimization for AMD NPU are in the in the [vitisai](./vitisai/) folder.

## **Optimization and Quantization for QUALCOMM NPU**

- [QUALCOMM NPU: PTQ + AOT Compilation using QNN EP](../phi3_5/README.md):
  - Run the workflow with `olive run --config qnn/qwen_2.5_1.5b_instruct_qnn_config.json`.
  - Run the inference with `python app.py -m models/qwen_2.5_1.5b_Instruct -c "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"`.
  - Config for Qwen 2.5 7b instruct model at qnn/qwen_2.5_7b_instruct_qnn_config.json
  - Run app.py with the correct chat template.
