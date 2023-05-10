# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-base) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Mixed Precision Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*

Outputs the final model and latency results.

**Important:** To run the example on Windows, please use cmd or PowerShell as administrator.

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Pip requirements
This example requires the latest code from onnxruntime and onnxruntime-extensions which are not available in the stable releases yet.
So, we will install the nightly versions.

On Linux:
```bash
# Install requirements
python -m pip install -r requirements.txt
# Install nightly versions of onnxruntime and onnxruntime-extensions
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
export OCOS_NO_OPENCV=1
python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git
```

On Windows (cmd):
```cmd
:: Install requirements
python -m pip install -r requirements.txt
:: Install nightly versions of onnxruntime and onnxruntime-extensions
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 onnxruntime-extensions==0.8.0.306180 ^
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

On Windows (PowerShell):
```powershell
# Install requirements
python -m pip install -r requirements.txt
# Install nightly versions of onnxruntime and onnxruntime-extensions
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 onnxruntime-extensions==0.8.0.306180 `
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

### Prepare workflow config json
```
python prepare_whisper_configs.py [--model_name MODEL_NAME] [--no_audio_decoder]

# For example, using whisper tiny model
python prepare_whisper_configs.py --model_name openai/whisper-tiny.en
```

`--model_name MODEL_NAME` is the name or path of the whisper model. The default value is `openai/whisper-tiny.en`.

`--no_audio_decoder` is optional. If not provided, will use audio decoder in the preprocessing ops.

**Note:** If `--no_audio_decoder` is provided, you need to install `librosa` package before running the optimization steps below.

```bash
python -m pip install librosa
```

## Run the config to optimize the model
First, install required packages according to passes.
```bash
python -m olive.workflows.run --config whisper_{device}_{precision}.json --setup

# For example, to install packages for CPU, INT8
python -m olive.workflows.run --config whisper_cpu_int8.json --setup
```

Then, optimize the model

On Linux:
```bash
python -m olive.workflows.run --config whisper_{device}_{precision}.json 2> /dev/null

# For example, to optimize CPU, INT8
python -m olive.workflows.run --config whisper_cpu_int8.json 2> /dev/null
```

On Windows (cmd):
```cmd
python -m olive.workflows.run --config whisper_{device}_{precision}.json 2> NUL

:: For example, to optimize CPU, INT8
python -m olive.workflows.run --config whisper_cpu_int8.json 2> NUL
```

On Windows (PowerShell):
```powershell
python -m olive.workflows.run --config whisper_{device}_{precision}.json 2> $null

# For example, to optimize CPU, INT8
python -m olive.workflows.run --config whisper_cpu_int8.json 2> $null
```

## Test the transcription of the optimized model
```bash
python test_transcription.py --config whisper_{device}_{precision}.json [--auto_path AUDIO_PATH]

# For example, to test CPU, INT8 with default audio path
python test_transcription.py --config whisper_cpu_int8.json
```

`--audio_path` is optional. If not provided, will use test auto path from the config.
