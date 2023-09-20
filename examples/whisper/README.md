# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-tiny) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Intel® Neural Compressor Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Mixed Precision Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*
- GPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops*

Outputs the final model and latency results.

**Important:** To run the example on Windows, please use cmd or PowerShell as administrator.

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

If you want to run the optimization pipeline with Intel® Neural Compressor, please make sure that `olive-ai[inc]` is installed.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

**Note:** Multilingual support requires ONNX Runtime 1.16.0+ which is not released yet. Must be built from or after commit https://github.com/microsoft/onnxruntime/commit/4b69226fca914753844a3291818ce23ac2f00d8c.

After running the above, uninstall pre-existing ONNX Runtime package and install the latest nightly build of ONNX Runtime as follows:
```bash
python -m pip uninstall -y onnxruntime ort-nightly
python -m pip install ort-nightly --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

### Prepare workflow config json
```
python prepare_whisper_configs.py [--model_name MODEL_NAME] [--no_audio_decoder] [--multilingual]

# For example, using whisper tiny model
python prepare_whisper_configs.py --model_name openai/whisper-tiny.en
```

`--model_name MODEL_NAME` is the name or path of the whisper model. The default value is `openai/whisper-tiny.en`.  
`--no_audio_decoder` is optional. If not provided, will use audio decoder in the preprocessing ops.

**Note:** If `--no_audio_decoder` is provided, you need to install `librosa` package before running the optimization steps below.

```bash
python -m pip install librosa
```

`--multiligual` is optional. If provided, the model produced will support multiple languages that are controlled using `decoder_input_ids` input.

**Example of decoder_input_ids:**
```python
import numpy as np
from transformers import AutoConfig, AutoProcessor


model = "openai/whisper-tiny"
config = AutoConfig.from_pretrained(model)
processor = AutoProcessor.from_pretrained(model)

# English transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
# forced_decoder_ids is of the format [(1, 50259), (2, 50359), (3, 50363)] and needs to be
# of the format [50258, 50259, 50359, 50363] where 50258 is the start token id
forced_decoder_ids = [config.decoder_start_token_id] + list(map(lambda token: token[1], forced_decoder_ids))

# If you don't want to provide specific decoder input ids or you want
# Whisper to predict the output language and task, you can set
# forced_decoder_ids = [config.decoder_start_token_id]
# [50258]

# decoder input ids
decoder_input_ids = np.array([forced_decoder_ids], dtype=np.int32)
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
python test_transcription.py --config whisper_{device}_{precision}.json [--auto_path AUDIO_PATH] [--language LANGUAGE] [--task {transcribe,translate}]

# For example, to test CPU, INT8 with default audio path
python test_transcription.py --config whisper_cpu_int8.json
```

`--audio_path` Optional. Path to audio file. If not provided, will use the test data from the config

`--language` Optional. Language spoken in audio. Default is `english`. Only used when `--multilingual` is provided to `prepare_whisper_configs.py`

`--task` Optional. Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate'). Default is `transcribe`. Only used
when `--multilingual` is provided to `prepare_whisper_configs.py`

## FAQ
The following are some common issues that may be encountered when running this example.
1. `INVALID_GRAPH / Error Node (BeamSearch_node) has input size 12 not in range [min=5, max=10]` or similar error when running inference using the optimized model.
This is likely due to:
    - Mismatch between the versions of onnxruntime used to run the optimization workflow and the inference. To fix this, please use the same version of
    onnxruntime for both.
    - Mismatch between the versions of onnxruntime used in a previously cached run and the currently installed version. To fix this, please delete the `cache` folder
    and run the workflow again.

2. Whenever you install a new version of onnxruntime (such as ort-nightly), you may need to delete the `cache` folder and run the workflow again. This is because the cache doesn't
distinguish between different versions of onnxruntime and will use the cached models from a previous run. There might be incompatibilities between the cached models and the new
version of onnxruntime.
