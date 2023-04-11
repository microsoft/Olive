# BERT optimization with PTQ on CPU
This is a sample use case of Olive to optimize a [Bert](https://huggingface.co/Intel/bert-base-uncased-mrpc) model using onnx conversion, onnx transformers optimization,
onnx quantization tuner and performance tuning.

Performs optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> Tune performance*

Outputs the best metrics, model, and corresponding Olive config.

## Prerequisites
Please go to example repository [Quickstart Bert Example](https://github.com/microsoft/Olive/tree/main/examples/bert_ptq_cpu)
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample using config. The optimization techniques to run are specified in bert_config.json
```
python -m olive.workflows.run --config bert_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_config.json")
```

## Optimize model automatically without selecting any optimization technique.
```
python -m olive.workflows.run --config auto_bert_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("auto_bert_config.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
Here is an example of Olive output(Footprints):
<!-- TODO -->