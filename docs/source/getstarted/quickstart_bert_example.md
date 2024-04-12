# BERT optimization with PTQ on CPU
This is a sample use case of Olive to optimize a [Bert](https://huggingface.co/Intel/bert-base-uncased-mrpc) model using onnx conversion, onnx transformers optimization,
onnx quantization tuner and performance tuning.

Performs optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model ->  ONNX Runtime performance tuning*

## Prerequisites
Please go to example repository [Quickstart Bert Example](https://github.com/microsoft/Olive/tree/main/examples/bert)
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample using config. The optimization techniques to run are specified in bert_ptq_cpu.json
First, install required packages according to passes.
```
olive run --config bert_ptq_cpu.json --setup
```
Then, optimize the model
```
olive run --config bert_ptq_cpu.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_ptq_cpu.json")
```

## Optimize model automatically without selecting any optimization technique.
First, install required packages according to passes.
```
olive run --config bert_auto.json --setup
```
Then, optimize the model
```
olive run --config bert_auto.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_auto.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
