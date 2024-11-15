# Quantize

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision. A quantized model executes some or all of the operations on tensors with reduced precision rather than full precision (floating point) values. This allows for a more compact model representation and the use of high performance vectorized operations on many hardware platforms.

OLIVE encapsulates all the latest cutting edge quantization techniques into a single command line tool that enables you to easily experiment/test the impact of different techniques.

## Supported quantization techniques

Currently, OLIVE supports the following techniques:

```{Note}
Some methods require a GPU and/or a calibration dataset.
```

| Method | Description | GPU required | Calibration dataset required | Input model format(s) | Output model format |
| ------ | ------------ | ------------ | ------------------ | ------------------ | ------------------- |
| AWQ | Activation-aware Weight Quantization (AWQ) creates 4-bit quantized models and it speeds up models by 3x and reduces memory requirements by 3x compared to FP16.  | ✔️ | ❌ | :simple-pytorch: <br> :simple-huggingface: | :simple-pytorch: |
| GPTQ | Generative Pre-trained Transformer Quantization (GPTQ) is a one-shot weight quantization method. You can quantize your favorite language model to 8, 4, 3 or even 2 bits.  | ✔️ | ✔️  | :simple-pytorch: <br> :simple-huggingface: |  :simple-pytorch:  |
| QuaRot | Quantization technique that combines quantization and rotation to reduce the number of bits required to represent the weights of a model.  | ✔️ | ✔️  | :simple-huggingface: |  :simple-pytorch:  |
| bnb4 | Is a MatMul with weight quantized with N bits (e.g., 2, 3, 4, 5, 6, 7). | ❌ | ❌ | :simple-onnx: | :simple-onnx: |
| ONNX Dynamic | Dynamic quantization calculates the quantization parameters (scale and zero point) for activations dynamically. | ❌ | ❌ | :simple-onnx: | :simple-onnx: |
| INC Dynamic | Intel® Neural Compressor model compression tool.  | ❌ | ❌ | :simple-onnx: | :simple-onnx: |
| NVMO | NVIDIA TensorRT Model Optimizer is a library comprising state-of-the-art model optimization techniques including quantization, sparsity, distillation, and pruning to compress models. | ❌ | ❌ | :simple-onnx: | :simple-onnx: |

## {octicon}`zap` Quickstart

To use AWQ quantization on [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) run the following command:


```{Note}
- You'll need to execute this command on a GPU machine.
- If you want to quantize a different model, update the `--model_name_or_path` to a different Hugging Face Repo ID (`{username}/{model})
```

```bash
olive quantize \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --algorithms awq \
    --output_path quantized-model \
    --log_level 1
```


## Quantization with ONNX Graph Capture

As articulated in [Supported quantization techniques](#supported-quantization-techniques), you may wish to take the PyTorch/Hugging Face output of AWQ/GPTQ/QuaRot quantization methods and convert into an ONNX format so that you can inference using the ONNX runtime. Alternatively, you may wish to capture the ONNX graph first and then run the model through bnb4/INC Dynamic/ONNX Dynamic/NVMO quantization methods.

To enable the conversion from PyTorch/Hugging Face format to ONNX, OLIVE provides a `capture-onnx-graph` command. The chain of commands required to quantize a model and output in a format for the ONNX runtime are:

```bash
# Step 1: AWQ (will output a PyTorch model)
olive quantize \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --algorithms awq \
    --output_path quantized-model \
    --log_level 1

# Step 2: Capture the ONNX graph
olive capture-onnx-graph \
    --model_name_or_path quantized-model/model \
    --use_ort_genai True \
    --log_level 1 \
```

## Pre-processing for Finetuning

Quantizing a model as a *pre*-processing step for finetuning rather than as a *post*-processing step leads to more accurate quantized models. The chain of OLIVE CLI commands required to quantize, finetune and output an ONNX model for the ONNX runtime are:

```bash
# Step 1: AWQ (will output a PyTorch model)
olive quantize \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --algorithms awq \
    --output_path quantized-model \
    --log_level 1

# Step 2: Finetune (will output a PEFT adapter)
olive finetune \
        --model_name_or_path quantized-model/model \
        --trust_remote_code \
        --output_path finetuned-model \
        --data_name xxyyzzz/phrase_classification \
        --text_template "<|start_header_id|>user<|end_header_id|>\n{phrase}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{tone}" \
        --method qlora \
        --max_steps 30 \
        --log_level 1 \

# Step 3: Generate Adapters for ONNX model (will output an ONNX Model)
olive generate-adapter \
    --model_name_or_path finetuned-model/model \
    --adapter_path finetuned-model/adapter \
    --use_ort_genai \
    --output_path adapter-onnx \
    --log_level 1
```

Once the `olive generate-adapter` has successfully completed, you'll have:

1. The base model in an optimized ONNX format.
2. The adapter weights in a format for ONNX Runtime.
