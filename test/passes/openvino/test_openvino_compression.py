# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path

from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.hardware import AcceleratorSpec
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.optimum_conversion import OptimumConversion
from olive.passes.openvino.compression import OpenVINOWeightCompression
from test.utils import get_hf_model


@Registry.register_dataset()
def wikitext_2_raw_v1_test():
    import datasets

    return datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")


def test_openvino_weight_compression_hf_to_openvino(tmp_path):
    # imports here
    import numpy as np
    from nncf.parameters import CompressWeightsMode, SensitivityMetric

    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-LlamaForCausalLM")

    def custom_transform_func(data, tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="np")
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1
        inputs["position_ids"] = position_ids

        batch_size = input_ids.shape[0]
        inputs["beam_idx"] = np.arange(batch_size, dtype=int)

        return inputs

    openvino_weight_compression_config = {
        "compress_config": {
            "mode": CompressWeightsMode.INT4_SYM,
            "ratio": 0.8,
            "sensitivity_metric": SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        },
        "transform_fn": custom_transform_func,
        "extra_args": {"tokenizer": True},
        "data_config": DataConfig(
            name="compress_data_config",
            load_dataset_config=DataComponentConfig(type="wikitext_2_raw_v1_test"),
        ),
    }
    p = create_pass_from_dict(
        OpenVINOWeightCompression,
        openvino_weight_compression_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_wc_hf_to_ov")

    # execute
    hf_to_ov_model = p.run(input_hf_model, output_folder)

    # define the XML and bin files paths if openvino models are produced
    xml_file = Path(hf_to_ov_model.model_path) / "openvino_model.xml"
    bin_file = Path(hf_to_ov_model.model_path) / "openvino_model.bin"

    # test if the model file is created
    assert xml_file.exists()
    assert xml_file.is_file()
    assert bin_file.exists()
    assert bin_file.is_file()

    # cleanup
    if Path(hf_to_ov_model.model_path).exists():
        shutil.rmtree(hf_to_ov_model.model_path)


def test_openvino_weight_compression_hf_to_onnx(tmp_path):
    from nncf.parameters import CompressWeightsMode

    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-LlamaForCausalLM")

    openvino_weight_compression_config = {
        "compress_config": {"mode": CompressWeightsMode.INT4_SYM, "ratio": 1.0, "all_layers": True},
        "extra_args": {"use_onnx": True, "advanced_compression_parameters": {"backend_params": {"external_dir": True}}},
        "ignored_scope": ["Gather"],
        "ignored_scope_type": "types",
    }
    p = create_pass_from_dict(
        OpenVINOWeightCompression,
        openvino_weight_compression_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_wc_hf_to_onnx")

    # execute
    hf_to_onnx_model = p.run(input_hf_model, output_folder)

    # test if the model file is created
    assert Path(hf_to_onnx_model.model_path).exists()
    assert Path(hf_to_onnx_model.model_path).is_file()

    # cleanup
    if Path(hf_to_onnx_model.model_path).is_file():
        q_dir = Path(hf_to_onnx_model.model_path).parent
    else:
        q_dir = Path(hf_to_onnx_model.model_path)
    shutil.rmtree(q_dir)


def test_openvino_weight_compression_onnx_to_onnx(tmp_path):
    from nncf.parameters import CompressWeightsMode

    # setup
    input_hf_model = get_hf_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
    optimum_conversion_config = {}
    p_optimum = create_pass_from_dict(
        OptimumConversion,
        optimum_conversion_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "CPUExecutionProvider"),
    )

    # create output folder
    output_folder_optimum = str(Path(tmp_path) / "optimum_convert")
    input_onnx_model = p_optimum.run(input_hf_model, output_folder_optimum)

    openvino_weight_compression_config = {
        "compress_config": {"mode": CompressWeightsMode.INT4_SYM, "ratio": 1.0, "all_layers": True},
        "extra_args": {"use_onnx": True, "advanced_compression_parameters": {"backend_params": {"external_dir": True}}},
        "ignored_scope": ["Gather"],
        "ignored_scope_type": "types",
    }
    p = create_pass_from_dict(
        OpenVINOWeightCompression,
        openvino_weight_compression_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )

    # create output folder
    output_folder = str(Path(tmp_path) / "openvino_wc_onnx_to_onnx")

    # execute
    onnx_to_onnx_model = p.run(input_onnx_model, output_folder)

    # test if the model file is created
    assert Path(onnx_to_onnx_model.model_path).exists()
    assert Path(onnx_to_onnx_model.model_path).is_file()

    # cleanup
    shutil.rmtree(output_folder_optimum)
    if Path(onnx_to_onnx_model.model_path).is_file():
        q_dir = Path(onnx_to_onnx_model.model_path).parent
    else:
        q_dir = Path(onnx_to_onnx_model.model_path)
    shutil.rmtree(q_dir)
