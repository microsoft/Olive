# Inception model optimization on Qualcomm NPU
This folder contains a sample use case of Olive to convert a TensorFlow model to SNPE DLC, quantize it and convert it to Onnx.

Performs optimization pipeline:
- *TensorFlow Model -> SNPE Model -> Quantized SNPE Model -> Onnx Model (for SNPE EP)*

Outputs a summary of the accuracy and latency metrics for each SNPE model.

## Prerequisites
### Download and unzip SNPE SDK
Download the SNPE SDK zip following [instructions from Qualcomm](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

We test it with SNPE v2.18.0.240101.

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`

### Configure SNPE
```sh
# in general, python 3.8 is recommended
olive-cli configure-qualcomm-sdk --py_version 3.8 --sdk snpe
# only when the tensorflow 1.15.0 is needed, use python 3.6
olive-cli configure-qualcomm-sdk --py_version 3.6 --sdk snpe
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

## Prepare the configuration file
The configuration file `inception_config.json` contains the parameters for the conversion and quantization.
But the quantization is only supported for `x64-Linux` platform. To run the optimization on Windows, please remove the `quantization` section from the configuration file.

Or you can just run following command to generate the configuration file:
```sh
python prepare_config.py
```

## Run sample
Run the conversion and quantization locally. Only supports `x64-Linux`.
```
olive-cli run --config inception_config.json
```
