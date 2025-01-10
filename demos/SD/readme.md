# SD to qnn
## Model

https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main

## Dependency

https://github.com/microsoft/Olive/issues/1267: `pip install transformers==4.42.4`

Need this version for textencoder

## Run

`python stable_diffusion.py --optimize`

## Bugs

### 1 debug shape infer

D:\Olive\olive-venv\Lib\site-packages\onnxruntime\quantization\quant_utils.py: load_model_with_shape_infer

```
    # Reload model instead
    if not model.opset_import or not model.ir_version:
        print("-"*20)
        print(inferred_model_path)
        print(model.opset_import)
        print(model.ir_version)
        print(len(model.graph.node))
        model = onnx.load(model_path.as_posix())
    else:
        inferred_model_path.unlink()
```

### 2 debug aug model

D:\Olive\olive-venv\Lib\site-packages\onnxruntime\quantization\calibrate.py: MinMaxCalibrater: augment_graph

```
# Output calibrated tensors
print("+"*20)
print(len(self.model.graph.node))
print(len(tensors))

# Save aug model
onnx.save(
    self.model,
    "aug.onnx",
    save_as_external_data=self.use_external_data_format,
)
```


## Other

https://github.com/quic/aimet/tree/develop
