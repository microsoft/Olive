# SNPE

The Snapdragon Neural Processing Engine (SNPE) is a Qualcomm Snapdragon software accelerated runtime for the execution of
deep neural networks.

Olive provides tools to convert models from different frameworks such as ONNX and TensorFlow to SNPE Deep Learning Container
(DLC) file and quantize them to 8 bit fixed point for running on the Hexagon DSP. Olive uses the development tools available
in the [Snapdragon Neural Processing Engine SDK](https://developer.qualcomm.com/sites/default/files/docs/snpe/index.html) also known as
[Qualcomm Neural Processing SDK for AI](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk).

## Prerequisites
### Download and unzip SNPE SDK
Download the SNPE SDK zip following [instructions from Qualcomm](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk).

We test it with SNPE v2.18.0.240101.

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`.

### Configure Olive SNPE
```bash
# in general, python 3.8 is recommended
olive configure-qualcomm-sdk --py_version 3.8 --sdk snpe
# only when the tensorflow 1.15.0 is needed, use python 3.6
olive configure-qualcomm-sdk --py_version 3.6 --sdk snpe
```
**Note:** If `olive` cannot be found in your path, you can use `python -m olive` instead.

## Model Conversion
`SNPEConversion` converts ONNX or TensorFlow models to SNPE DLC. The DLC file can be loaded into the SNPE runtime for inference
using one of the Snapdragon accelerated compute cores.

Please refer to [SNPEConversion](snpe_conversion) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "SNPEConversion",
    "input_names": ["input"],
    "input_shapes": [[1, 299, 299, 3]],
    "output_names": ["InceptionV3/Predictions/Reshape_1"],
    "output_shapes": [[1, 1001]]
}
```

## Post Training Quantization (PTQ)
`SNPEQuantization` quantizes the DLC file. Quantized DLC files use fixed point representations of network parameters,
generally 8 bit weights and 8 or 32bit biases. Please refer to the
[corresponding documentation](https://developer.qualcomm.com/sites/default/files/docs/snpe/quantized_models.html) for more
details.

Please refer to [SNPEQuantization](snpe_quantization) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "SNPEQuantization",
    "data_dir": "data_dir",
    "user_script": "user_script.py",
    "dataloader_func": "create_quant_dataloader",
    "enable_htp": true
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/inception/user_script.py)
for an example implementation of `"user_script.py"` and `"create_quant_dataloader"`.
