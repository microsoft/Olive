# Quantize

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision. A quantized model executes some or all of the operations on tensors with reduced precision rather than full precision (floating point) values. This allows for a more compact model representation and the use of high performance vectorized operations on many hardware platforms.

Olive encapsulates all the latest cutting edge quantization techniques into a single command line tool that enables you to easily experiment/test the impact of different techniques.

## Supported quantization techniques

Currently, Olive supports the following techniques:

```{Note}
Some methods require a GPU and/or a calibration dataset.
```

| Implementation | Description | Model format(s) | Algorithm | GPU required |
| -------------- | ----------- | --------------- | --------- | ------------ |
| AWQ | Activation-aware Weight Quantization (AWQ) creates 4-bit quantized models and it speeds up models by 3x and reduces memory requirements by 3x compared to FP16.  | PyTorch <br> ONNX| Awq | ✔️ |
| GPTQ | Generative Pre-trained Transformer Quantization (GPTQ) is a one-shot weight quantization method. You can quantize your favorite language model to 8, 4, 3 or even 2 bits.  | PyTorch <br> ONNX |  GptQ  | ✔️ |
| BitsAndBytes | Is a MatMul with weight quantized with N bits (e.g., 2, 3, 4, 5, 6, 7). | ONNX | RTN | ❌ |
| ORT | Static and dynamic quantizations. | ONNX | RTN | ❌ |
| INC | Intel® Neural Compressor model compression tool. | ONNX | GPTQ | ❌ |
| NVMO | NVIDIA TensorRT Model Optimizer is a library comprising state-of-the-art model optimization techniques including quantization, sparsity, distillation, and pruning to compress models. | ONNX | AWQ | ❌ |

## {octicon}`zap` Quickstart

To use AWQ quantization on [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) run the following command:


```{Note}
- You'll need to execute this command on a GPU machine.
- If you want to quantize a different model, update the `--model_name_or_path` to a different Hugging Face Repo ID (`{username}/{model})
```

```bash
olive quantize \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --algorithm awq \
    --output_path models/llama/awq \
    --log_level 1
```


## Quantization with ONNX Optimizations

As articulated in [Supported quantization techniques](#supported-quantization-techniques), you may wish to take the PyTorch/Hugging Face output of AWQ/GPTQ quantization methods and convert into an optimized ONNX format so that you can inference using the ONNX runtime.

You can use Olive's automatic optimizer (`auto-opt`) to create an optimized ONNX model from a quantized model:

```bash
# Step 1: AWQ (will output a PyTorch model)
olive quantize \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --algorithm awq \
    --output_path models/llama/awq \
    --log_level 1

# Step 2: Create an optimized ONNX model
olive auto-opt \
   --model_name_or_path models/llama/awq \
   --device cpu \
   --provider CPUExecutionProvider \
   --use_ort_genai \
   --output_path models/llama/onnx \
   --log_level 1
```

## Pre-processing for Finetuning

Quantizing a model as a *pre*-processing step for finetuning rather than as a *post*-processing step leads to more accurate quantized models because the loss due to quantization can be recovered during fine-tuning. The chain of Olive CLI commands required to quantize, finetune and output an ONNX model for the ONNX runtime are:

```bash
# Step 1: AWQ (will output a PyTorch model)
olive quantize \
   --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
   --trust_remote_code \
   --algorithm awq \
   --output_path models/llama/awq \
   --log_level 1

# Step 2: Finetune (will output a PEFT adapter)
olive finetune \
    --method lora \
    --model_name_or_path models/llama/awq \
    --data_name xxyyzzz/phrase_classification \
    --text_template "<|start_header_id|>user<|end_header_id|>\n{phrase}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{tone}" \
    --max_steps 100 \
    --output_path ./models/llama/ft \
    --log_level 1

# Step 3: Optimized ONNX model (will output an ONNX Model)
olive auto-opt \
   --model_name_or_path models/llama/ft/model \
   --adapter_path models/llama/ft/adapter \
   --device cpu \
   --provider CPUExecutionProvider \
   --use_ort_genai \
   --output_path models/llama/onnx \
   --log_level 1
```

Once the automatic optimizer has successfully completed, you'll have:

1. The base model in an optimized ONNX format.
2. The adapter weights in a format for ONNX Runtime.
