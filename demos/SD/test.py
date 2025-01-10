import onnxruntime as ort
import numpy as np
import onnx
from onnx import shape_inference, TensorProto

file = "./footprints/unet/output_model/model/model.onnx"
file_new = "new.onnx"
file_infer = "infer.onnx"

# Inference
if False:
    unet = ort.InferenceSession(file)

    batch_size = 1
    unet_sample_size = 64
    cross_attention_dim = 1024
    torch_dtype = np.float32

    inputs = {
        "latent": np.random.rand(batch_size, 4, unet_sample_size, unet_sample_size).astype(np.float32),
        "time_emb": np.random.rand(batch_size).astype(np.float32),
        "text_emb": np.random.rand(batch_size, 77, cross_attention_dim).astype(np.float32),
    }

    result = unet.run(None, inputs)

    print(result)

# Add tensor_value_info
if False:
# https://blog.csdn.net/Artyze/article/details/107403191

    ONNX_DTYPE = {
        0: TensorProto.FLOAT,
        1: TensorProto.FLOAT,
        2: TensorProto.UINT8,
        3: TensorProto.INT8,
        4: TensorProto.UINT16,
        5: TensorProto.INT16,
        6: TensorProto.INT32,
        7: TensorProto.INT64,
        8: TensorProto.STRING,
        9: TensorProto.BOOL
    }

    onnx_model = onnx.load(file)
    print(onnx_model.ir_version)
    #print(onnx_model.graph.input)
    #print(onnx_model.graph.initializer)

    tensor = onnx_model.graph.input[0]
    tensor0 = onnx.helper.make_tensor_value_info(tensor.name, 1, [1,4,64,64])
    onnx_model.graph.input.remove(tensor)

    tensor = onnx_model.graph.input[0]
    tensor1 = onnx.helper.make_tensor_value_info(tensor.name, 1, [1])
    onnx_model.graph.input.remove(tensor)

    tensor = onnx_model.graph.input[0]
    tensor2 = onnx.helper.make_tensor_value_info(tensor.name, 1, [1,77,1024])
    onnx_model.graph.input.remove(tensor)

    onnx_model.graph.input.append(tensor0)
    onnx_model.graph.input.append(tensor1)
    onnx_model.graph.input.append(tensor2)

    onnx.save(onnx_model, file_new)
    onnx.checker.check_model(file_new)

# https://github.com/onnx/onnx/issues/6150
if False:
    onnx_model = onnx.load(file, load_external_data=False)
    onnx_infer = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_infer, file_infer)
