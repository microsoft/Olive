# Llama 3 Optimization

Sample use cases of Olive to optimize [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model using Olive.

- [Quantize, Finetune and Optimize for CPU/CUDA](../getting_started/olive-awq-ft-llama.ipynb)
- [QDQ Model with 4-bit Weights & 16-bit Activations](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config.json -m meta-llama/Llama-3.2-1B-Instruct -o models/llama3-qdq`.
- [AMD NPU: Optimization and Quantization with for VitisAI](../phi3_5/README.md):
  - Run the workflow with `olive run --config qdq_config_vitis_ai.json -m meta-llama/Llama-3.2-1B-Instruct -o models/llama3-vai`.
- [PTQ + AOT Compilation for Qualcomm NPUs using QNN EP](../phi3_5/README.md):
  - Run the workflow with `olive run --config qnn_config.json -m meta-llama/Llama-3.2-1B-Instruct -o models/llama3-qnn`.
  - Run the inference with `python app.py -m models/llama3-qnn -c "<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"`.
- [PTQ + AWQ ONNX OVIR Encapsulated 4-bit weight compression using Optimum OpenVINO](./openvino/)

**NOTE:**

- Access to the [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

- The quality of the quantized model is not guaranteed to be the same as the original model, especially for such a small model. Work is ongoing to improve the quality of the quantized model.
