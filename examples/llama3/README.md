# Llama 3 Optimization

Sample use cases of Olive to optimize [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model using Olive.

- [Quantize, Finetune and Optimize for CPU/CUDA](../getting_started/olive-awq-ft-llama.ipynb)
- [QDQ Model with 4-bit Weights & 16-bit Activations](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config.json -m meta-llama/Llama-3.2-1B-Instruct -o models/llama3-qdq`.
- [AMD NPU: Optimization and Quantization with for VitisAI](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config_vitis_ai.json -m meta-llama/Llama-3.2-1B-Instruct -o models/llama3-vai`.
- [QUALCOMM NPU: PTQ + AOT Compilation using QNN EP](../phi3_5/README.md):
  - Refer to the Qualcomm NPU section below.
- [PTQ + AWQ ONNX OVIR Encapsulated 4-bit weight compression using Optimum OpenVINO](./openvino/)

**NOTE:**

- Access to the [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

- The quality of the quantized model is not guaranteed to be the same as the original model, especially for such a small model. Work is ongoing to improve the quality of the quantized model.

## **Optimization and Quantization for AMD NPU**

- [**AMD NPU**](./vitisai/): Instructions to run quantization and optimization for AMD NPU are in the in the [vitisai](./vitisai/) folder.

## **Optimization and Quantization for QUALCOMM NPU**

- [QUALCOMM NPU: PTQ + AOT Compilation using QNN EP](../phi3_5/README.md):
  - Run the workflow with `olive run --config qnn/llama3.2_1b_instruct_qnn_config.json`.
  - Run the inference with `python app.py -m models/llama_3.2_1b -c "<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"`.
  - Config for llama 3.1 8b instruct model at qnn/llama3.1_8b_instruct_qnn_config.json
  - Run app.py with the correct chat template.
