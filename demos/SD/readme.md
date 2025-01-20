# SD to qnn
## Model

https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main

### Text encoder

Constant: 263
Reshape: 233
Gather: 5
ConstantOfShape: 1
Cast: 1
Mul: 25
Add: 209
ArgMax: 1
Equal: 1
LayerNormalization: 47
Where: 1
MatMul: 184
Expand: 1
Transpose: 115
Softmax: 23
Gelu: 23
Shape: 2
Flatten: 1
Concat: 2

1138 -> 370

### Unet

Constant: 649
Conv: 66
MatMul: 256
ConstantOfShape: 1
Reshape: 282
Shape: 109
Mul: 222
InstanceNormalization: 61
Transpose: 160
Equal: 1
Where: 1
Expand: 1
Add: 281
Unsqueeze: 45
Sigmoid: 47
Cast: 98
Sin: 1
Cos: 1
Concat: 14
Slice: 66
Gemm: 24
Div: 102
LayerNormalization: 48
Sqrt: 96
Softmax: 32
Gather: 16
Gelu: 16
Resize: 3

2699 -> 680

About .opt

This is for CUDA

File "D:\Olive\olive-venv\Lib\site-packages\onnx\shape_inference.py", line 96, in infer_shapes_path
    C.infer_shapes_path(model_path, output_path, check_type, strict_mode, data_prop)
onnx.onnx_cpp2py_export.shape_inference.InferenceError: [TypeInferenceError] Cannot infer type and shape for node name GroupNorm_0. No opset import for domain com.microsoft optype GroupNorm

GroupNorm is not supported https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html

### Vae decoder

Conv: 36
Constant: 117
Reshape: 66
InstanceNormalization: 30
Shape: 31
Mul: 61
Add: 49
Sigmoid: 29
Div: 16
Transpose: 6
MatMul: 6
Slice: 1
Cast: 4
Sqrt: 3
Softmax: 1
Resize: 3

459 -> 70

### About Op types

From Renyi (also working): "MatMul", "LayerNormalization", "Gemm", "Sigmoid"

From Renyi (not working): Add, Mul

Constant: no impact for decoder

Conv: big size reduce for decoder, but not work

## Run

### Dependency

https://github.com/microsoft/Olive/issues/1267: `pip install transformers==4.42.4`

Need this version for textencoder

### Convert to onnx

ONNX model must use an opset >= 17 in order to use LayerNormalization

`python stable_diffusion.py --model [text_encoder/unet/vae_decoder]`

### Use auto-opt

olive auto-opt --model_name_or_path "D:\Olive\demos\SD\footprints\unet\output_model\model\model.onnx" --output_path footprints/unet_auto --device cpu --provider CPUExecutionProvider --precision int16 --log_level 1

### Use OnnxStaticQuantization

`python stable_diffusion.py --model [text_encoder/unet/vae_decoder] --qnn`

### Use QNN SDK

See D:\Olive\examples\mobilenet\README_QNN_SDK.md

```
export QNN_SDK_ROOT=/mnt/c/Qualcomm/AIStack/QAIRT/2.28.0.241029

# See D:\Olive\olive\platform_sdk\qualcomm\configure\configure.py
# Need to update to 3.10?
# file:///C:/Qualcomm/AIStack/QAIRT/2.28.0.241029/docs/QNN/general/setup/linux_setup.html said so
# Not needed PIP_EXTRA_ARGS
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
from collections import Counter
type_counts = Counter([node.op_type for node in self.model.graph.node])
for string, count in type_counts.items():
    print(f"{string}: {count}")
print(len(tensors))

# Save aug model
onnx.save(
    self.model,
    "aug.onnx",
    save_as_external_data=self.use_external_data_format,
)
```

### 3 op_types_to_quantize is overwitten by prepare_qnn_config

D:\Olive\olive-venv\Lib\site-packages\olive\passes\onnx\quantization.py: OnnxQuantization: _run_for_config

```
                # if we set it, overwrite from qnn_qdq_config
                if config["op_types_to_quantize"]:
                    run_config["op_types_to_quantize"] = config["op_types_to_quantize"]
                print("^"*20)
                print(run_config)
```

### 4 inference with local system?

D:\Olive\olive-venv\Lib\site-packages\onnxruntime\quantization\calibrate.py

CalibraterBase.infer_session if use CPUExecutionProvider, do not support float16

## Other

https://github.com/quic/aimet/tree/develop

https://app.aihub.qualcomm.com/docs/hub/quantize_examples.html

