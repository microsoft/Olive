## Prepare model
python -m onnxruntime.transformers.convert_generation -m gpt2 --model_type gpt2 --output ./models/onnx_models/gpt2_beam_search.onnx --cache_dir ./models
python -m onnxruntime.transformers.convert_generation -m gpt2 -p fp16 --use_gpu --model_type gpt2 --output ./gpt2_fp16/onnx_models/gpt2_beam_search.onnx


## CPU
```
pip uninstall onnxruntime
pip install onnxruntime==1.13.1
python -m olive.workflows.run --config cpu_config.json
```

## GPU
```
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.13.1
python -m olive.workflows.run --config gpu_config.json
```


## Notes when construct user_script.py
1. there may contains multi metrics. How we return them for one metric config.
