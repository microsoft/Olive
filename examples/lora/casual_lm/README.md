# Fine tuning LM model using LoRA

THIS IS A WORK IN PROGRESS, INCOMPLETE EXAMPLE

This folder contains a sample use case of Olive to finetune and optimize a PyTorch model using LoRA.

Performs optimization pipeline:
- *PyTorch Model -> LoRA -> FineTune -> Onnx Model ->*

```
python -m olive.workflows.run --config lora_casual_lm.json
```
