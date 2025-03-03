## How to run
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Prepare models
```
python prepare_onnx_model.py
```

### Run sample using config
```
olive run --config ./face_yolo_qnn.json
```

**Note**: The special configuration of op_types_to_quantize in the face_yolo_qnn.json file is to exclude the mul operation. This is because after quantizing the mul operation, the latency of this model on the QNN will increase significantly.

