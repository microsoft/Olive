# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

from inception_utils import get_directories
from user_script import create_quant_dataloader

from olive.model import TensorFlowModel
from olive.passes import SNPEConversion, SNPEQuantization, SNPEtoONNXConversion
from olive.passes.olive_pass import create_pass_from_dict
from olive.snpe import SNPEDevice


def get_args():
    parser = argparse.ArgumentParser(description="Olive vnext Inception example")

    args = parser.parse_args()
    return args


def main():
    # args = get_args()

    _, models_dir, data_dir = get_directories()
    name = "inception_v3"

    # ------------------------------------------------------------------
    # Tensorflow model
    print("Loading Tensorflow model...")
    tensorflow_model_file = str(models_dir / f"{name}.pb")
    tensorflow_model = TensorFlowModel(model_path=tensorflow_model_file)

    # ------------------------------------------------------------------
    # SNPE model
    print("Converting Tensorflow model to SNPE...")
    snpe_model_file = str(models_dir / f"{name}_snpe.dlc")

    snpe_conversion = create_pass_from_dict(
        SNPEConversion,
        {
            "input_names": ["input"],
            "input_shapes": [[1, 299, 299, 3]],
            "output_names": ["InceptionV3/Predictions/Reshape_1"],
            "output_shapes": [[1, 1001]],
        },
        disable_search=True,
    )
    snpe_model = snpe_conversion.run(tensorflow_model, snpe_model_file)
    assert Path(snpe_model.model_path).is_file()
    json.dump(
        snpe_model.io_config,
        open(str(models_dir / f"{Path(snpe_model_file).stem}.io_config.json"), "w"),
        indent=2,
    )

    # ------------------------------------------------------------------
    # SNPE Quantized model
    print("Quantizing SNPE model...")
    snpe_quantized_model_file = str(models_dir / f"{name}_snpe_quantized.dlc")
    snpe_quantized_data_dir = str(data_dir)
    snpe_quantization = create_pass_from_dict(
        SNPEQuantization,
        {"data_dir": snpe_quantized_data_dir, "dataloader_func": create_quant_dataloader, "enable_htp": True},
        disable_search=True,
    )
    snpe_quantized_model = snpe_quantization.run(snpe_model, snpe_quantized_model_file)
    assert Path(snpe_quantized_model.model_path).is_file()
    json.dump(
        snpe_quantized_model.io_config,
        open(str(models_dir / f"{Path(snpe_quantized_model_file).stem}.io_config.json"), "w"),
        indent=2,
    )

    # ------------------------------------------------------------------
    # SNPE Quantized ONNX model
    print("Converting SNPE Quantized model to ONNX...")
    snpe_quantized_onnx_model_file = str(models_dir / f"{name}_snpe_quantized.onnx")

    snpe_to_onnx_conversion = create_pass_from_dict(
        SNPEtoONNXConversion, {"target_device": SNPEDevice.DSP}, disable_search=True
    )
    snpe_quantized_onnx_model = snpe_to_onnx_conversion.run(snpe_quantized_model, snpe_quantized_onnx_model_file)
    assert Path(snpe_quantized_onnx_model.model_path).is_file()


if __name__ == "__main__":
    main()
