# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-base) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Mixed Precision Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*

Outputs the best metrics, model, and corresponding Olive config.

## Prerequisites
### Pip requirements
First ensure that Olive is installed in your environment.

This example requires the latest code from onnxruntime and onnxruntime-extensions which are not available in the stable releases yet. So, we
will install the nightly versions.

On Linux:
```bash
python -m pip install -r requirements.txt
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
export OCOS_NO_OPENCV=1
python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git
```

On Windows:
```bash
python -m pip install -r requirements.txt
python -m pip uninstall -y onnxruntime onnxruntime-extensions
python -m pip install ort-nightly==1.15.0.dev20230429003 onnxruntime-extensions==0.8.0.303816 ^
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

### Prepare workflow config json
```
python prepare_configs.py [--model_name MODEL_NAME]
```

`model_name` is the name or path of the whisper model. The default value is `openai/whisper-base.en`.


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
