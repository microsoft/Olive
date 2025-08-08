# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import platform
from pathlib import Path

import pytest
import torch
from onnxscript import ir
from packaging import version

from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.onnxscript_fusion import OnnxScriptFusion


# TODO(anyone): Remove the skip condition once the issue is resolved
@pytest.mark.skipif(platform.system() == "Windows", reason="Skip on Windows due to export failure")
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.7.0"), reason="Requires PyTorch 2.7 or higher")
def test_onnxscript_fusion_pass_works(tmp_path):
    base_model = HfModelHandler(model_path="katuni4ka/tiny-random-phi3")
    conversion_pass = create_pass_from_dict(
        OnnxConversion, {"torch_dtype": "float32", "use_dynamo_exporter": True}, disable_search=True
    )
    output_folder = str(tmp_path / "onnx")
    onnx_model = conversion_pass.run(base_model, output_folder)

    assert Path(onnx_model.model_path).exists()
    onnx_model_ir = ir.load(onnx_model.model_path)
    onnx_model_nodes = list(onnx_model_ir.graph)

    # inplace fusion
    fusion_pass = create_pass_from_dict(OnnxScriptFusion, {}, disable_search=True)
    fusion_model = fusion_pass.run(onnx_model, output_folder)

    assert Path(fusion_model.model_path).exists()
    fusion_model_ir = ir.load(fusion_model.model_path)
    fusion_model_nodes = list(fusion_model_ir.graph)

    assert len(onnx_model_nodes) > len(fusion_model_nodes)
    assert not any(n.domain == "com.microsoft" for n in onnx_model_nodes)
    assert any(n.domain == "com.microsoft" for n in fusion_model_nodes)
