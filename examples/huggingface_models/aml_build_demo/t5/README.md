## Prepare model
python -m onnxruntime.transformers.convert_generation -m t5-small --model_type t5 --output ./models/t5/onnx_models/t5_small_beam_search.onnx --cache_dir ./models
python -m onnxruntime.transformers.convert_generation -m t5-small -p fp16 --use_gpu --model_type t5 --output ./models/t5/onnx_models_fp16/t5_small_beam_search.onnx --cache_dir ./models


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
