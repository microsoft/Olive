import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        batch_size = 1
        unet_sample_size = 64
        cross_attention_dim = 1024
        torch_dtype = np.float32

        inputs = {
            "latent": np.random.rand(batch_size, 4, unet_sample_size, unet_sample_size).astype(np.float32),
            "time_emb": np.random.rand(batch_size).astype(np.float32),
            "text_emb": np.random.rand(batch_size, 77, cross_attention_dim).astype(np.float32),
        }

        self.data_list = []
        self.data_list.append(inputs)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


input_model_path = "D:\\Olive\\demos\\SD\\footprints\\unet\\output_model\\model\\model.onnx"  # TODO: Replace with your actual model
output_model_path = "model.qdq.onnx"  # Name of final quantized model
my_data_reader = DataReader(input_model_path)

# Pre-process the original float32 model.
preproc_model_path = "model.preproc.onnx"
model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)
print(model_changed)
model_to_quantize = preproc_model_path if model_changed else input_model_path

# Generate a suitable quantization configuration for this model.
# Note that we're choosing to use uint16 activations and uint8 weights.
qnn_config = get_qnn_qdq_config(model_to_quantize,
                                my_data_reader,
                                activation_type=QuantType.QUInt16,  # uint16 activations
                                weight_type=QuantType.QUInt8)       # uint8 weights

# Quantize the model.
quantize(model_to_quantize, output_model_path, qnn_config)
