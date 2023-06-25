# BERT Optimization
This folder contains examples of BERT optimization using different workflows.

## Optimization Workflows
### BERT optimization with PTQ on CPU
This workflow performs BERT optimization on CPU with ONNX Runtime PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

This workflow also demonstrates how to use:
- Huggingface `transformers` to load model from [model hub](https://huggingface.co/models).
- Huggingface `datasets` to load data from [dataset hub](https://huggingface.co/datasets).
- Huggingface `evaluate` to load multi metrics from [metric hub](https://huggingface.co/evaluate-metric).

Config file: [bert_ptq_cpu.json](bert_ptq_cpu.json)

#### AzureML Model Source and No Auto-tuning
The workflow in [bert_ptq_cpu_aml.json](bert_ptq_cpu_aml.json) is similar to the above workflow, but uses AzureML Model Source to load the model and does not perform auto-tuning. Without auto-tuning, the passes will be run with the default parameters (no search space) and the final model and metrics will be saved in the output directory.

In order to use this example, `<place_holder>` in the `azureml_client` section must be replaced with the appropriate values for your
AzureML workspace.


### BERT optimization with Intel® Neural Compressor PTQ on CPU
This workflow performs BERT optimization on CPU with Intel® Neural Compressor quantization tuner. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Intel® Neural Compressor Quantized Onnx Model*

Config file: [bert_inc_ptq_cpu.json](bert_inc_ptq_cpu.json)

#### Run Intel® Neural Compressor quantization with or without accuracy aware tuning

Accuracy aware tuning is one of unique features provided by Intel® Neural Compressor quantization. This feature can be used to solve accuracy loss pain points brought by applying low precision quantization and other lossy optimization methods. Intel® Neural Compressor also supports to quantize all quantizable ops without accuracy tuning, user can decide whether to tune the model accuracy or not. Please check the [doc](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md) for more details.

User can decide to tune the model accuracy by setting accuracy metric with goal in `evaluator`, and then setting `evaluator` in Intel® Neural Compressor quantization pass. If not set, accuracy of the model will not be tuned.

```json
"evaluators": {
    "common_evaluator": {
        "metrics":[
            {
                "name": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "percent-max-degradation", "value": 2}}
                ]
            }
        ]
    }
},
"passes": {
    "quantization": {
        "type": "IncQuantization",
        "config": {
                "evaluator": "common_evaluator"
            }
    }
}

```

#### Static Quantization
The workflow in [bert_inc_static_ptq_cpu.json](bert_inc_static_ptq_cpu.json) is similar to the above workflow, but specifically uses static quantization instead of static/dynamic quantization.
> **Note**: Custom accuracy metric is used in [bert_inc_static_ptq_cpu.json](bert_inc_static_ptq_cpu.json).

#### Dynamic Quantization
The workflow in [bert_inc_dynamic_ptq_cpu.json](bert_inc_dynamic_ptq_cpu.json) is similar to the above workflow, but specifically uses dynamic quantization instead of static/dynamic quantization.

### BERT optimization with QAT Customized Training Loop on CPU
This workflow performs BERT optimization on CPU with QAT Customized Training Loop. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> Transformers Optimized Onnx Model -> ONNX Runtime performance tuning*

Config file: [bert_qat_customized_train_loop_cpu.json](bert_qat_customized_train_loop_cpu.json)

### BERT optimization with CUDA/TensorRT
This workflow performs BERT optimization on GPU with CUDA/TensorRT. It performs the optimization pipeline:
1. CUDA: `CUDAExecutionProvider`
    - *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model with fp16 -> ONNX Runtime performance tuning*
    Config file: [bert_cuda_gpu.json](bert_cuda_gpu.json)
2. TensorRT: `TensorrtExecutionProvider`
    - *PyTorch Model -> Onnx Model -> ONNX Runtime performance tuning with trt_fp16_enable*
    Config file: [bert_trt_gpu.json](bert_trt_gpu.json)
## How to run
### Pip requirements
Install the necessary python packages:
```
[CPU]
python -m pip install -r requirements.txt
[GPU]
python -m pip install -r requirements-gpu.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```
python -m olive.workflows.run --config <config_file>.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config <config_file>.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
