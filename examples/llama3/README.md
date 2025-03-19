# Llama 3 Optimization

Sample use cases of Olive to optimize a [Llama 3.2 1B Instruct](meta-llama/Llama-3.2-1B-Instruct) model using Olive.
- [Quantize, Finetune and Optimize for CPU/CUDA](../getting_started/olive-awq-ft-llama.ipynb)
- [Optimize for Qualcomm NPU](../phi3_5/README.md): Replace `model_path` in `config.json` with `meta-llama/Llama-3.2-1B-Instruct`.

**NOTE:**
- Access to the model is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```
- The quality of the quantized model is not guaranteed to be the same as the original model, especially for such a small model. Work is ongoing to improve the quality of the quantized model.
