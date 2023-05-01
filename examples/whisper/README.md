# Whisper optimization using ORT toolchain

    ⚠️ THIS SAMPLE IS A WORK IN PROGRESS AND REQUIRES LATEST ONNXRUNTIME CODE (NOT YET RELEASED) ⚠️

This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-base) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Mixed Precision Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*
- GPU, INT8: *PyTorch Model -> Onnx Model -> Dynamic Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*

Outputs the best metrics, model, and corresponding Olive config.

**Note**: The template is not complete yet. There is no evaluator or search. Mixed precision, beam search, pre/post ops and tuning are not present.

## Prerequisites
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
