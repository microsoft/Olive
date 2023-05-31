# BERT Optimization
This folder contains examples of BERT optimization using different workflows.

## Optimization Workflows
### BERT optimization with PTQ on CPU
This workflow performs BERT optimization on CPU with PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> Tune performance*

Config file: [bert_ptq_cpu.json](bert_ptq_cpu.json)

#### AzureML Model Source and No auto-tuning
The workflow in [bert_hf_cpu_aml.json](bert_hf_cpu_aml.json) is similar to the above workflow, but uses AzureML Model Source to load the model and does not perform auto-tuning. Without auto-tuning, the passes will be run with the default parameters (no search space) and the final model and metrics will be saved in the output directory.

In order to use this example, the `<place_holder>`s in the `azureml_client` section must be replaced with the appropriate values for your
AzureML workspace.

### BERT optimization with Intel® Neural Compressor PTQ on CPU
This workflow performs BERT optimization on CPU with Intel® Neural Compressor quantization tuner. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Intel® Neural Compressor Quantized Onnx Model

Config file: [bert_inc_ptq_cpu.json](bert_inc_ptq_cpu.json)

### BERT optimization with QAT Customized Training Loop on CPU
This workflow performs BERT optimization on CPU with QAT Customized Training Loop. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> Transformers Optimized Onnx Model -> ONNX Runtime performance tuning*

Config file: [bert_qat_customized_train_loop_cpu.json](bert_qat_customized_train_loop_cpu.json)

## How to run
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Run sample using config.

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
