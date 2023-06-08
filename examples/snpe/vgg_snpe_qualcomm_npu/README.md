# VGG model optimization on Qualcomm NPU
This folder contains a sample use case of Olive to convert an Onnx model to SNPE DLC, quantize it and convert it to Onnx.

Performs optimization pipeline:
- *Onnx Model -> SNPE Model -> Quantized SNPE Model*

## Prerequisites
### Download and unzip SNPE SDK
Download the SNPE SDK zip following [instructions from Qualcomm](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`.

### Configure SNPE
```
python -m olive.snpe.configure
```

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Download data and model
To download the necessary data and model files:
```
python download_files.py
```

## Run sample
Run the conversion and quantization locally. Only supports `x64-Linux`.
```
python -m olive.workflows.run --config vgg_config.json
```
