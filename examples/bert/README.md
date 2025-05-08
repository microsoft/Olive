# BERT Optimization
This folder contains examples of BERT optimization using different workflows.

- QDQ: [Int8 Quantization with QDQ format](#bert-quantization-qdq)
- CPU: [Optimization with PTQ for model from HF/AML](#bert-optimization-with-ptq-on-cpu)
- CPU: [Optimization with Intel® Neural Compressor PTQ](#bert-optimization-with-intel®-neural-compressor-ptq-on-cpu)
- CPU: [Optimization with QAT Customized Training Loop](#bert-optimization-with-qat-customized-training-loop-on-cpu)
- GPU: [Optimization with CUDA/TensorRT](#bert-optimization-with-cudatensorrt-on-gpu)
- Qualcomm NPU: [Optimization with PTQ on Qualcomm NPU using QNN EP](./qnn/)
- Intel® NPU: [Optimization with OpenVINO on Intel® NPU to generate an ONNX OpenVINO IR Encapsulated Model](./openvino/)
- AMD NPU: [Optimization and Quantization with QDQ format for AMD NPU (VitisAI)](#optimization-and-quantization-for-amd-npu)

Go to [How to run](#how-to-run)


## Optimization Workflows
### BERT Quantization QDQ
 This workflow quantizes the model. It performs the pipeline:
 - *HF Model-> ONNX Model ->Quantized Onnx Model*

 Config file: [Intel/bert-base-uncased](bert_ptq_qdq.json)

 #### Accuracy / Latency / Throughput

 | Model Version         | Accuracy (Top-1)    | Latency (ms/sample)  | Throughput (token per second)| Dataset   |
 |-----------------------|---------------------|----------------------|------------------------------|-----------|
 | PyTorch FP32          | 90%                 | 2406                 | 0.41                         | glue-mrpc |
 | ONNX INT8 (QDQ)       | 90%                 | 401                  | 2.51                         | glue-mrpc |

 *Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*

 Config file: [google-bert/bert-base-multilingual-cased](google_bert_qdq.json)

 #### Latency / Throughput

 | Model Version         | Latency (ms/sample)  | Throughput (token per second)| Dataset       |
 |-----------------------|----------------------|------------------------------|---------------|
 | PyTorch FP32          | 1162                 | 0.81                         | facebook/xnli |
 | ONNX INT8 (QDQ)       | 590                  | 1.75                         | facebook/xnli |

### Optimization and Quantization for AMD NPU

 This workflow quantizes the model. It performs the pipeline:
 - *HF Model-> ONNX Model -> Optimizations -> Quantized Onnx Model*

 Config files for VitisAI:
 - [Intel/bert-base-uncased](bert_ptq_qdq_vitis_ai.json)
 - [google-bert/bert-base-multilingual-cased](google_bert_qdq_vitis_ai.json)


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

User can decide to tune the model accuracy by setting an accuracy `metric` in Intel® Neural Compressor quantization pass config. If not set, accuracy of the model will not be tuned.

```json
"passes": {
    "quantization": {
        "type": "IncQuantization",
        "metric": {
            "name": "accuracy",
            "type": "accuracy",
            "sub_types": [
                {"name": "accuracy_score", "priority": 1, "goal": {"type": "percent-max-degradation", "value": 2}}
            ]
        }
    }
}

```

#### Static Quantization
The workflow in [bert_inc_static_ptq_cpu.json](bert_inc_static_ptq_cpu.json) is similar to the above workflow, but specifically uses static quantization instead of static/dynamic quantization.
> **Note**: Custom accuracy metric is used in [bert_inc_static_ptq_cpu.json](bert_inc_static_ptq_cpu.json).

#### Dynamic Quantization
The workflow in [bert_inc_dynamic_ptq_cpu.json](bert_inc_dynamic_ptq_cpu.json) is similar to the above workflow, but specifically uses dynamic quantization instead of static/dynamic quantization.

#### Run with SmoothQuant

Quantizing activations in large language models (LLMs) with huge parameter sizes can be challenging due to the presence of outliers. The SmoothQuant method, introduced in this [paper](https://arxiv.org/abs/2211.10438), addresses this issue by transferring the quantization difficulty from activations to weights through a mathematically equivalent transformation by using a fixed-value $\alpha$ for the entire model. However, the distributions of activation outliers vary not only across different models but also across different layers within a model. To resolve this, Intel® Neural Compressor proposes a method to obtain layer-wise optimal $\alpha$ values with the ability to tune automatically. Please refer to this [link](https://github.com/intel/neural-compressor/blob/master/docs/source/smooth_quant.md) for more algorithm details.

User can use SmoothQuant by setting `smooth_quant` in `recipes` as shown below. Refer to [bert_inc_smoothquant_ptq_cpu.json](bert_inc_smoothquant_ptq_cpu.json) for an example of SmoothQuant.

```json
"passes": {
    "quantization": {
        "type": "IncStaticQuantization",
        "recipes":{
            "smooth_quant": true,
            "smooth_quant_args": {"alpha": 0.5}
        }
    }
}
```

### BERT optimization with QAT Customized Training Loop on CPU
This workflow performs BERT optimization on CPU with QAT Customized Training Loop. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> Transformers Optimized Onnx Model -> ONNX Runtime performance tuning*

Config file: [bert_qat_customized_train_loop_cpu.json](bert_qat_customized_train_loop_cpu.json)

### BERT optimization with CUDA/TensorRT on GPU
This workflow performs BERT optimization on GPU with CUDA/TensorRT. It performs the optimization pipeline:
1. CUDA: `CUDAExecutionProvider`
    - *PyTorch Model -> Onnx Model -> ONNX Runtime performance tuning*
    Run: [bert.py](bert.py)
    - *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model with fp16 -> ONNX Runtime performance tuning*
    Run: [bert.py](bert.py) --optimize
2. TensorRT: `TensorrtExecutionProvider`
    - *PyTorch Model -> Onnx Model -> ONNX Runtime performance tuning with trt_fp16_enable*
    Config file: [bert_trt_gpu.json](bert_trt_gpu.json)

## How to run
### Pip requirements
Install the necessary python packages:
```sh
# [CPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[cpu]
# [GPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[gpu]
# [NPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[qnn]
```

# Other dependencies
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```sh
olive run --config <config_file>.json --setup
```

Then, optimize the model
```sh
olive run --config <config_file>.json
```

After running the above command, the final model will be saved in the *output_dir* specified in the config file.
