# SD to qnn
## Model

https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main

## Dependency

https://github.com/microsoft/Olive/issues/1267: `pip install transformers==4.42.4`

Need this version for textencoder

## Run

### Use OnnxStaticQuantization

`python stable_diffusion.py --model [text_encoder/unet/vae_decoder] --qnn`

### Use QNN SDK

See D:\Olive\examples\mobilenet\README_QNN_SDK.md

```
export QNN_SDK_ROOT=/mnt/c/Qualcomm/AIStack/QAIRT/2.28.0.241029

# See D:\Olive\olive\platform_sdk\qualcomm\configure\configure.py
bash /mnt/d/Olive/olive/platform_sdk/qualcomm/create_python_env.sh -v 3.8 --sdk qnn

# Fix pandas build
sudo ln -s /mnt/c/Qualcomm/AIStack/QAIRT/2.28.0.241029/python-env-setup/olive-pyenv/lib/python3.10/site-packages/numpy/core/include/numpy /usr/include/numpy

# Fix "onnx/onnx_ml_pb2.py", line 5, in <module> from google.protobuf.internal import builder as _builder ImportError: cannot import name 'builder' from 'google.protobuf.internal'
/mnt/c/Qualcomm/AIStack/QAIRT/2.28.0.241029/olive-pyenv/bin/python -m pip install --upgrade protobuf

olive run --config config_vae_decoder.qnn.sdk.json
```


## Bugs

### 1 shape infer does not support > 2GB

https://github.com/onnx/onnx/issues/6150

D:\Olive\olive-venv\Lib\site-packages\onnxruntime\quantization\quant_utils.py: load_model_with_shape_infer

```
    # Use model without data instead
    if not model.opset_import or not model.ir_version:
        print("-"*20)
        print(inferred_model_path)

        model = onnx.load(model_path.as_posix())
        ext_model_path = generate_identified_filename(model_path, "-ext")
        onnx.save_model(model, ext_model_path.as_posix(), save_as_external_data=True)

        model = onnx.load(ext_model_path.as_posix(), load_external_data=False)
        onnx_infer = onnx.shape_inference.infer_shapes(model)
        onnx.save(onnx_infer, inferred_model_path)

        model = onnx.load(inferred_model_path.as_posix())
        add_infer_metadata(model)
        print(len(model.graph.node))
    else:
        inferred_model_path.unlink()
```

### 2 debug aug model

not needed, but useful to show how many tensors are quantized

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

https://app.aihub.qualcomm.com/docs/hub/quantize_examples.html

