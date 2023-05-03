# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-base) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Mixed Precision Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*

Outputs the final model and latency results.

## Prerequisites
### Pip requirements
This example requires the latest code from onnxruntime and onnxruntime-extensions which are not available in the stable releases yet. So, we
will install the nightly versions.

We recommending using a fresh conda or virtual environment for this example to avoid any conflicts with your existing environment.

On Linux:
```bash
# Create a new conda environment
conda create -n olive-whisper python=3.8
conda activate olive-whisper
# Install Olive from source
python -m pip install git+https://github.com/microsoft/Olive
# Install requirements
python -m pip install -r requirements.txt
# Install nightly versions of onnxruntime and onnxruntime-extensions
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
export OCOS_NO_OPENCV=1
python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git
```

On Windows:
```cmd
:: Create a new conda environment
conda create -n olive-whisper python=3.8
conda activate olive-whisper
:: Install Olive from source
python -m pip install git+https://github.com/microsoft/Olive
:: Install requirements
python -m pip install -r requirements.txt
:: Install nightly versions of onnxruntime and onnxruntime-extensions
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 onnxruntime-extensions==0.8.0.303816 ^
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

**Note:** You can also use a python virtual environment instead of conda.

On Linux:
```bash
# Create a new virtual environment
python -m venv olive-whisper
source olive-whisper/bin/activate
```

On Windows:
```cmd
:: Create a new virtual environment
python -m venv olive-whisper
olive-whisper\Scripts\activate.bat
```

### Prepare workflow config json
```
python prepare_configs.py [--model_name MODEL_NAME] [--no_audio_decoder]
```

`--model_name MODEL_NAME` is the name or path of the whisper model. The default value is `openai/whisper-base.en`.

`--no_audio_decoder` is optional. If not provided, will use audio decoder in the preprocessing ops.

## To optimize Whisper model run the sample config
First, install required packages according to passes.
```
python -m olive.workflows.run --config whisper_{device}_{precision}.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config whisper_{device}_{precision}.json
```

## Test the transcription of the optimized model
```
python test_transcription.py --config whisper_{device}_{precision}.json [--auto_path AUDIO_PATH]
```

`--audio_path` is optional. If not provide, will use test auto path from the config.
cda
