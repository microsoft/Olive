## Prepare model
git clone https://github.com/microsoft/onnxruntime
cd ./onnxruntime/python/tools/transformers/models/bart/
python export.py -m facebook/bart-large-cnn -o ./models/


Config:
max_length: 20
min_length: 0
input_text: None
spm_path: None
vocab_path: None
num_beams: 5
repetition_penalty: 1.0
no_repeat_ngram_size: 3
early_stopping: False
opset_version: 14

model_dir: facebook/bart-large-cnn

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
