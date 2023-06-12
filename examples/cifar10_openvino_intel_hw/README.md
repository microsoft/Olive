# CIFAR10 optimization with OpenVINO for Intel HW
This folder contains a sample use case of Olive to optimize a CIFAR10 model using OpenVINO conversion and OpenVINO quantization.

Performs optimization pipeline:
- *PyTorch Model -> OpenVINO Model -> Quantized OpenVINO Model*

Outputs a summary of the accuracy and latency metrics for each model.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample
### Local evaluation
Evaluate the models locally:
```
python cifar10.py
```
