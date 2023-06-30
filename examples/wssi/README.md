# WSSI Unified Workflow Example

This is an example of a WSSI workflow that uses the olive for model conversion-quantization.

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

## Convert-Quantize
```bash
python convertquantize.py --config model_config.json --tool {snpe, openvino}
```
