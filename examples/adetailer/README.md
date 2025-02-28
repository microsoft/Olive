## How to run
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Prepare data and model
```
python prepare_onnx_model.py
```

### Run sample using config

olive run --config ./face_yolo_qnn.json