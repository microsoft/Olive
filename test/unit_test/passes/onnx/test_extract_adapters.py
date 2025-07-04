# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnxruntime.quantization.calibrate import CalibrationDataReader
from packaging import version
from peft import LoHaConfig, LoraConfig, get_peft_model
from peft.tuners.loha import LoHaLayer
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

from olive.common.utils import WeightsFileFormat, find_submodules, load_weights
from olive.model import HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import AdapterType, model_has_adapters
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.extract_adapters import ExtractAdapters
from olive.passes.onnx.rtn_quantization import OnnxBlockWiseRtnQuantization
from test.unit_test.utils import get_onnx_model


class LlamaCalibrationDataLoader(CalibrationDataReader):
    # pylint: disable=W0223
    def __init__(self, dummy_input):
        super().__init__()
        self.count = 0
        self.max_count = 10
        self.dummy_input = {k: v.numpy() for k, v in dummy_input.items()}

    def get_next(self):
        if self.count >= self.max_count:
            return None

        self.count += 1
        return self.dummy_input

    def rewind(self):
        self.count = 0


def get_calib_data_loader(dummy_input):
    return LlamaCalibrationDataLoader(dummy_input)


@pytest.fixture(name="input_model_info", scope="module")
def input_model_info_fixture(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp("extract-adapters-test")

    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    adapter_type = request.param
    use_dora = adapter_type == AdapterType.DORA

    pytorch_model = AutoModelForCausalLM.from_pretrained(model_name)
    # init_lora_weights are set so that lora_B weights are not all zeros
    # if they are all zeros, the exported onnx model uses identity node as input to lora_B

    # keep track of all lora modules
    if adapter_type == AdapterType.LOHA:
        peft_model = get_peft_model(pytorch_model, LoHaConfig(init_weights=False, target_modules="all-linear"))
        all_modules = [
            m.replace("base_model.model.", "") for m in find_submodules(peft_model, LoHaLayer, full_name=True) or []
        ]
        adapter_suffix = ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]
    else:
        peft_model = get_peft_model(pytorch_model, LoraConfig(init_lora_weights=False, use_dora=use_dora))
        all_modules = [
            m.replace("base_model.model.", "") for m in find_submodules(peft_model, LoraLayer, full_name=True) or []
        ]
        adapter_suffix = ["dora_A", "dora_B", "dora_M"] if use_dora else ["lora_A", "lora_B"]

    # names of float weights
    all_weights = [f"{m}.{suffix}.weight" for m in all_modules for suffix in adapter_suffix]
    all_quant_weights = [
        w.replace(".weight", suffix)
        for w in all_weights
        for suffix in [".quant.weight", ".quant.scale", ".quant.zero_point"]
        if not w.endswith("M.weight")  # Dora Div is not quantized
    ]
    if adapter_type == AdapterType.DORA:
        all_quant_weights += [f"{m}.dora_M.weight" for m in all_modules]
    if adapter_type == AdapterType.LOHA:
        int4_weight = [
            "hada_w1_a.default",
            "hada_w1_b.default_Q4",
            "hada_w1_b.default_scales",
            "hada_w2_a.default",
            "hada_w2_b.default_Q4",
            "hada_w2_b.default_scales",
        ]
        weights = [f"{m}.{suffix.replace('default', 'weight')}" for m in all_modules for suffix in int4_weight]
        all_quant_weights = [w + ".quant" for w in weights]

    # dump adapters
    adapters_path = tmp_path / "pytorch-adapters"
    peft_model.save_pretrained(adapters_path)
    del peft_model, pytorch_model

    # pytorch model
    olive_pytorch_model = HfModelHandler(model_path=model_name, task="text-generation", adapter_path=adapters_path)

    # export to onnx
    conversion_pass = create_pass_from_dict(OnnxConversion, {"target_opset": 14}, disable_search=True)
    olive_onnx_model = conversion_pass.run(olive_pytorch_model, str(tmp_path / "onnx-export"))

    # TODO(jambayk): re-enable qdq model test once flaky quantization failure is resolved
    # static QDQ quantization
    # qdq_pass = create_pass_from_dict(
    #     OnnxStaticQuantization,
    #     {
    #         "dataloader_func": lambda data_dir, batch_size: LlamaCalibrationDataLoader(
    #             olive_pytorch_model.get_dummy_inputs()
    #         )
    #     },
    #     disable_search=True,
    # )
    # olive_qdq_onnx_model = qdq_pass.run(olive_onnx_model, str(tmp_path / "qdq-onnx"))

    # int4 quantization
    rtn_quantizer = create_pass_from_dict(OnnxBlockWiseRtnQuantization, {}, disable_search=True)
    olive_int4_onnx_model = rtn_quantizer.run(olive_onnx_model, str(tmp_path / "int4-onnx"))

    return {
        "adapter_type": adapter_type,
        "float": {"onnx_model": olive_onnx_model, "all_weights": all_weights},
        # "qdq": {
        #     "onnx_model": olive_qdq_onnx_model,
        #     "all_weights": all_quant_weights
        # },
        "int4": {
            "onnx_model": olive_int4_onnx_model,
            "all_weights": {name for name in all_quant_weights if "zero_point" not in name},
        },
        "adapter_path": adapters_path,
    }


@pytest.mark.parametrize("input_model_info", [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA], indirect=True)
@pytest.mark.parametrize("model_type", [None, "float", "int4"])
def test_model_has_adapters(input_model_info, model_type):
    model_info = input_model_info
    adapter_type = model_info["adapter_type"]

    if model_type is None:
        assert not model_has_adapters(get_onnx_model().model_path, adapter_type)
    else:
        assert model_has_adapters(model_info[model_type]["onnx_model"].model_path, adapter_type)


@pytest.mark.parametrize("input_model_info", [AdapterType.LORA], indirect=True)
@pytest.mark.parametrize("quantize_int4", [1, 0])
@pytest.mark.parametrize("adapter_format", [el.value for el in WeightsFileFormat])
def test_convert_adapters_command(tmp_path, adapter_format, quantize_int4, input_model_info):
    if adapter_format == WeightsFileFormat.ONNX_ADAPTER and version.parse(ort.__version__) < version.parse("1.20"):
        pytest.skip("ONNX_ADAPTER format is only supported in onnxruntime 1.20+")

    from olive.cli.launcher import main as cli_main

    # args
    suffix = ".npz" if adapter_format == WeightsFileFormat.NUMPY else f".{adapter_format}"
    exported_adapters_path = tmp_path / f"exported-adapters.{suffix}"
    args = [
        "convert-adapters",
        "--adapter_path",
        str(input_model_info["adapter_path"]),
        "--output_path",
        str(exported_adapters_path),
        "--adapter_format",
        str(adapter_format),
    ]
    if quantize_int4:
        args.append("--quantize_int4")

    # execute
    cli_main(args)

    # assert
    assert Path(exported_adapters_path).is_file()
    weight_dtype = "int4" if quantize_int4 else "float"
    assert set(input_model_info[weight_dtype]["all_weights"]) == set(load_weights(exported_adapters_path))


@pytest.mark.parametrize("input_model_info", [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA], indirect=True)
@pytest.mark.parametrize("model_type", ["float", "qdq", "int4"])
def test_extract_adapters(tmp_path, model_type, input_model_info):
    if model_type == "qdq":
        pytest.skip("QDQ model test is disabled due to flaky quantization failure")
    adapter_type = input_model_info["adapter_type"]
    if adapter_type in (AdapterType.DORA, AdapterType.LOHA) and model_type == "int4":
        pytest.skip("DORA/LoHa model test is disabled for int4 model")

    pass_config = {"make_inputs": False, "save_format": WeightsFileFormat.NUMPY, "adapter_type": adapter_type}

    p = create_pass_from_dict(ExtractAdapters, pass_config, disable_search=True)
    output_folder = tmp_path / "extracted-adapters"
    extracted_model: ONNXModelHandler = p.run(input_model_info[model_type]["onnx_model"], output_folder)

    # assert
    assert Path(extracted_model.model_path).is_file()
    assert Path(extracted_model.external_initializers_path).is_file()
    # all lora weights should be extracted as external initializers
    expected_weights = set(input_model_info[model_type]["all_weights"])
    assert expected_weights == set(extracted_model.model_attributes["external_initializers"])
    assert expected_weights == set(np.load(extracted_model.external_initializers_path))
    # ensure all external initializers are marked as such
    model_proto = onnx.load(extracted_model.model_path, load_external_data=False)
    seen_weights = set()
    for initializer in model_proto.graph.initializer:
        if initializer.name in expected_weights:
            assert initializer.data_location == onnx.TensorProto.EXTERNAL
            seen_weights.add(initializer.name)
    assert seen_weights == expected_weights


@pytest.mark.parametrize("input_model_info", [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA], indirect=True)
@pytest.mark.parametrize("model_type", ["float", "qdq", "int4"])
@pytest.mark.parametrize("save_format", [el.value for el in WeightsFileFormat])
def test_extract_adapters_as_inputs(tmp_path, save_format, model_type, input_model_info):
    if model_type == "qdq":
        pytest.skip("QDQ model test is disabled due to flaky quantization failure")
    if save_format == WeightsFileFormat.ONNX_ADAPTER and version.parse(ort.__version__) < version.parse("1.20"):
        pytest.skip("ONNX_ADAPTER format is only supported in onnxruntime 1.20+")
    adapter_type = input_model_info["adapter_type"]
    if adapter_type in (AdapterType.DORA, AdapterType.LOHA) and model_type == "int4":
        pytest.skip("DORA/LoHa model test is disabled for int4 model")

    # Create the configuration for the pass
    pass_config = {"save_format": save_format, "adapter_type": adapter_type}
    p = create_pass_from_dict(ExtractAdapters, pass_config, disable_search=True)
    output_folder = tmp_path / "extracted-adapters"

    # Execute the pass
    extracted_model: ONNXModelHandler = p.run(input_model_info[model_type]["onnx_model"], output_folder)
    io_config = extracted_model.io_config

    # Assertions
    assert Path(extracted_model.model_path).is_file()
    assert Path(extracted_model.constant_inputs_path).is_file()

    expected_weights = set(input_model_info[model_type]["all_weights"])

    # Check if all expected weights are extracted
    assert expected_weights == set(extracted_model.model_attributes["constant_inputs"])
    assert expected_weights == set(load_weights(extracted_model.constant_inputs_path))

    # Ensure all constant inputs are present in the input names
    assert all(i in io_config["input_names"] for i in expected_weights)
