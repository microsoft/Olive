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
