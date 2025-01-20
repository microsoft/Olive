input_model_path = "D:\\Olive\\demos\\SD\\footprints\\vae_decoder\\output_model\\model.onnx"
infer_model_path = "D:\\Olive\\demos\\SD\\footprints\\vae_decoder_infer\\output_model\\model.onnx"
output_model_path = "D:\\Olive\\demos\\SD\\footprints\\vae_decoder_onnx\\output_model\\model.onnx"
qnn_model_path = "D:\\Olive\\demos\\SD\\footprints\\vae_decoder_qnn\\output_model\\model.onnx"

# https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
# Preprocess

if False:
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(input_model_path, infer_model_path, skip_optimization=True, save_as_external_data=True, all_tensors_to_one_file=True)
    exit(0)

# quantize static

import os

folders = [f"data/{f.name}" for f in os.scandir('data') if f.is_dir()]
folders = folders[:1]
print(folders)

import numpy as np
from onnxruntime.quantization import CalibrationDataReader

class BaseDataLoader(CalibrationDataReader):
    def __init__(self):
        self.data = []
        self.idx = 0

    def get_next(self):
        print("getitem: " + str(self.idx))
        if self.idx >= len(self.data): return None
        self.idx += 1
        return self.data[self.idx - 1]

    def rewind(self):
        self.idx = 0


class UnetDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        for f in folders:
            for i in range(5):
                latent = np.fromfile(f + f'/{i}_latent.raw', dtype=np.float32).reshape(1, 4, 64, 64)
                time = np.fromfile(f + f'/{i}_time.raw', dtype=np.float32).reshape(1)
                text = np.fromfile(f + f'/{i}_text.raw', dtype=np.float32).reshape(1, 77, 1024)
                self.data.append({ "latent": latent, "time_emb": time, "text_emb": text })
                text = np.fromfile(f + f'/{i}_untext.raw', dtype=np.float32).reshape(1, 77, 1024)
                self.data.append({ "latent": latent, "time_emb": time, "text_emb": text })

class DecoderDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        for f in folders:
            data = np.fromfile(f + '/latent.raw', dtype=np.float32).reshape(1, 4, 64, 64)
            self.data.append({ "latent": data })

if False:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    dr = UnetDataLoader()
    quantize_static(
        infer_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt16,
        use_external_data_format=True,
    )
    exit(0)

# debug

if True:
    from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)

    def _generate_aug_model_path(model_path: str) -> str:
        aug_model_path = (
            model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
        )
        return aug_model_path + ".save_tensors.onnx"

    float_model_path = input_model_path
    qdq_model_path = qnn_model_path

    print("------------------------------------------------\n")
    print("Comparing weights of float model vs qdq model.....")

    matched_weights = create_weight_matching(float_model_path, qdq_model_path)
    weights_error = compute_weight_error(matched_weights)
    for weight_name, err in weights_error.items():
        print(f"Cross model error of '{weight_name}': {err}\n")

    print("------------------------------------------------\n")
    print("Augmenting models to save intermediate activations......")

    aug_float_model_path = _generate_aug_model_path(float_model_path)
    modify_model_output_intermediate_tensors(float_model_path, aug_float_model_path)

    aug_qdq_model_path = _generate_aug_model_path(qdq_model_path)
    modify_model_output_intermediate_tensors(qdq_model_path, aug_qdq_model_path)

    print("------------------------------------------------\n")
    print("Running the augmented floating point model to collect activations......")
    input_data_reader = DecoderDataLoader()
    float_activations = collect_activations(aug_float_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Running the augmented qdq model to collect activations......")
    input_data_reader.rewind()
    qdq_activations = collect_activations(aug_qdq_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Comparing activations of float model vs qdq model......")

    act_matching = create_activation_matching(qdq_activations, float_activations)
    act_error = compute_activation_error(act_matching)
    for act_name, err in act_error.items():
        print(f"Cross model error of '{act_name}': {err['xmodel_err']} \n")
        print(f"QDQ error of '{act_name}': {err['qdq_err']} \n")
    exit(0)
