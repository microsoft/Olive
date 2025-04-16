# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.optimum_intel import OpenVINOOptimumConversion
from test.unit_test.utils import get_hf_model

def test_openvino_optimum_conversion_pass_convert(tmp_path):
    # setup
    input_hf_model = get_hf_model()
    openvino_optimum_conversion_config = { "extra_args" : { "disable_convert_tokenizer": True } }

    p = create_pass_from_dict(OpenVINOOptimumConversion, openvino_optimum_conversion_config, disable_search=True)

    # create output folder
    output_folder = str(tmp_path / "openvino_optimum_convert")

    # execute
    ov_output_model = p.run(input_hf_model, output_folder)

    # define the XML and BIN file paths if openvino models are produced
    xml_file = Path(ov_output_model.model_path) / "openvino_model.xml"
    bin_file = Path(ov_output_model.model_path) / "openvino_model.bin"

    # test if the model xml and bin files are created
    assert xml_file.exists()
    assert xml_file.is_file()
    assert bin_file.exists()
    assert bin_file.is_file()
