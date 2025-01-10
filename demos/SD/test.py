import onnxruntime as ort
import numpy as np

unet = ort.InferenceSession("footprints/unet/output_model/model/model.onnx")

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
