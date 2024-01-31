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

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`.

```{note}
The SNPE SDK development environment is limited to Ubuntu, specifically version 18.04. It might not work as expected on Ubuntu 20.04. We
recommend using a Ubuntu 18.04 docker container if you don't have a machine running the same OS.
```

### Install SDK system dependencies
```bash
source $SNPE_ROOT/bin/dependencies.sh
```

### Configure Olive SNPE
```
python -m olive.snpe.configure
```

## Model Conversion
`SNPEConversion` converts ONNX or TensorFlow models to SNPE DLC. The DLC file can be loaded into the SNPE runtime for inference
using one of the Snapdragon accelerated compute cores.

Please refer to [SNPEConversion](snpe_conversion) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "SNPEConversion",
    "config": {
        "input_names": ["input"],
        "input_shapes": [[1, 299, 299, 3]],
        "output_names": ["InceptionV3/Predictions/Reshape_1"],
        "output_shapes": [[1, 1001]],
    }
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
    "config":  {
        "data_dir": "data_dir",
        "user_script": "user_script.py",
        "dataloader_func": "create_quant_dataloader",
        "enable_htp": true
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/inception/user_script.py)
for an example implementation of `"user_script.py"` and `"create_quant_dataloader"`.
