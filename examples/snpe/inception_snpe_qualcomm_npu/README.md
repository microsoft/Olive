# Inception model optimization on Qualcomm NPU
This folder contains a sample use case of Olive to convert a TensorFlow model to SNPE DLC, quantize it and convert it to Onnx.

Performs optimization pipeline:
- *TensorFlow Model -> SNPE Model -> Quantized SNPE Model -> Onnx Model (for SNPE EP)*

Outputs a summary of the accuracy and latency metrics for each SNPE model.

## Prerequisites
### Download and unzip SNPE SDK
Download the SNPE SDK zip following [instructions from Qualcomm](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`

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

## Run conversion and quantization
### Run locally
Run the conversions locally. Only supports `x86_64 Linux`.
```
python inception_dev.py
```

## Run evaluation of SNPE models (Can be run without the conversion and quantization step)
### Download data and model
If running evaluation on a Windows device, please ensure the snpe models from the conversion and quantization step
are saved in the models folder.

### Run locally
Evaluate the SNPE models locally. Supports `x86_64 Linux`, `x86_64 Windows` and `aarch64 Windows`.
```
python inception_eval.py [--use_dsp]
```

Note: `--use_dsp` only works on `aarch64 Windows`.
