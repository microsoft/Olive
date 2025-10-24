# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from test.utils import get_hf_model


def test_whisper_tiny_en(tmp_path):
    # setup
    from olive.passes.onnx.model_builder import ModelBuilder

    pytorch_model = get_hf_model("openai/whisper-tiny.en")
    onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
        pytorch_model, tmp_path/ "onnx_model"
    )
    print(f"ONNX model created at: {onnx_model.model_path}")
