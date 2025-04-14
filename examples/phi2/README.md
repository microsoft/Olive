# phi2 optimization with Olive
This folder contains an example of phi2 optimization with Olive workflow.

- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

## Prerequisites
* einops
* Pytorch>=2.2.0 \
  _The [official website](https://pytorch.org/) offers packages compatible with CUDA 11.8 and 12.1. Please select the appropriate version according to your needs._
* [ONNXRuntime nightly package](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages)
  In Linux, phi2 optimization requires the ONNXRuntime nightly package(>=1.18.0). In Windows, ONNXRuntime>=1.17.0 is recommended.

## Fine-tune phi2 Model using QLoRA
This workflow fine-tunes [phi2 model](https://huggingface.co/microsoft/phi-2) using [QLoRA](https://arxiv.org/abs/2305.14314) to generate text with given prompt.

You need to install required packages according to qlora. Also we suggest to use gpu devices for fine-tune process.
```bash
pip install -r requirements-lora.txt
```

Then, you can run the fine-tune using the following command:
```bash
python phi2.py --finetune_method qlora
```
Note that, to demonstrate the fine-tune process, we use a small training steps and a small dataset. For better performance, you can increase the training steps and use a larger dataset by updating
`phi2_optimize_template.json`.
We will consider to expose more parameters in the future to make it easier to customize the training process.

## Optimization Usage
In this stage, we will use the `phi2.py` script to generate optimized models and do inference with the optimized models.

Following are the model types that can be used for optimization:
cpu_fp32
```bash
# optimize the fine-tuned model
python phi2.py --finetune_method qlora --model_type cpu_fp32
# optimize the original model
```
cpu_int4
```bash
python phi2.py --model_type cpu_int4
```
cuda_fp16
```bash
python phi2.py --model_type cuda_fp16
```
cuda_int4
```bash
python phi2.py --model_type cuda_int4
```

### GenAI Optimization
For using ONNX runtime GenAI to optimize, follow build and installation instructions [here](https://github.com/microsoft/onnxruntime-genai) to install onnxruntime-genai package(>0.1.0).

Run the following command to execute the workflow:
```bash
olive run --config phi2_genai.json
```
This `phi2_genai.json` config file will generate optimized models for `cpu_int4` and `cuda_int4` model types as onnxruntime-gpu support cpu ep and cuda ep both.
If you only want cpu or cuda model, you can modify the config file by remove the unwanted execution providers.
```json
# CPU
"accelerators": [
  {
      "device": "CPU",
      "execution_providers": [
          "CPUExecutionProvider",
      ]
  }
]
# CPU: this is same with above as onnxruntime-gpu support cpu ep
"accelerators": [
  {
      "device": "GPU",
      "execution_providers": [
          "CPUExecutionProvider",
      ]
  }
]
# CUDA
"accelerators": [
  {
      "device": "GPU",
      "execution_providers": [
          "CUDAExecutionProvider",
      ]
  }
]
```

or you can use `phi2.py` to generate optimized models separately by running the following commands:
```bash
python phi2.py --model_type cpu_int4 --genai_optimization
python phi2.py --model_type cuda_int4 --genai_optimization
```

Snippet below shows an example run of generated phi2 model.
```python
import onnxruntime_genai as og

model = og.Model("model_path")
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=200)
generator = og.Generator(model, params)
generator.append_tokens(tokens)

while not generator.is_done():
    generator.generate_next_token()

text = tokenizer.decode(generator.get_sequence(0))

print("Output:")
print(text)
```

Also you can use `--inference` argument to run inference with the optimized model.
```bash
python phi2.py --model_type cuda_int4 --genai_optimization --inference
```

### Optimum Optimization
Above commands will generate optimized models with given model_type and save them in the `phi2` folder. These optimized models can be wrapped by ONNXRuntime for inference.
Besides, for better generation experience, this example also let use use [Optimum](https://huggingface.co/docs/optimum/v1.2.1/en/onnxruntime/modeling_ort) to generate optimized models.
Then use can call `model.generate` easily to run inference with the optimized model.
```bash
# optimum optimization. Please avoid to use optimum for fine-tune model which is not supported by now in Olive.
python phi2.py --model_type cpu_fp32 --optimum_optimization
```

Then let us use the optimized model to do inference.

## Generation example of optimized model
```bash
# --prompt is optional, can accept a string or a list of strings
# if not given, the default prompt "Write a function to print 1 to n" "Write a extremely long story starting with once upon a time"
python phi2.py --model_type cpu_fp32 --inference --prompt "Write a extremely long story starting with once upon a time"
```
This command will
1. generate optimized models if you never run the command before,
2. reuse the optimized models if you have run the command before,
3. then use the optimized model to do inference with greedy Top1 search strategy.
Note that, we only use the simplest greedy Top1 search strategy for inference example which may show not very reasonable results.

For better generation experience, here is the way to run inference with the optimized model using Optimum.
```bash
python phi2.py --model_type cpu_fp32 --inference --optimum_optimization --prompt "Write a extremely long story starting with once upon a time"
```

## Export output models in MLFlow format
If you want to output the optimized models to a zip file in MLFlow format, add `--export_mlflow_format` argument. The MLFlow model will be packaged in a zip file named `mlflow_model` in the output folder.

## Limitations
1. The latest ONNXRuntime implements specific fusion patterns for better performance but only works for ONNX model from TorchDynamo-based ONNX Exporter. And the TorchDynamo-based ONNX Exporter is only available on Linux.
When using Windows, this example will fallback to the default PyTorch ONNX Exporter, that can achieve a few improvements but not as much as the TorchDynamo-based ONNX Exporter.
Therefore, it is recommended to use Linux for phi2 optimization.

2. For Optimum optimization, the dynamo model is not supported very well. So we use legacy Pytorch ONNX Exporter to run optimization like what we do in Windows.

## Transformer Compression with SliceGPT
This workflow compresses a model to improve performance and reduce memory footprint. Specific details about the algorithm can be found in the linked [paper](https://arxiv.org/abs/2401.15024).

## Prerequisites
[slicegpt](https://github.com/microsoft/TransformerCompression)

To run the workflow,
```bash
python phi2.py --slicegpt
```
