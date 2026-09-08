# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from collections import Counter

import onnx
import onnx_ir as ir

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries


def run_surgery(model, tmp_path, surgeon, name):
    model_path = tmp_path / f"{name}.onnx"
    onnx.save(model, model_path)
    graph_surgeries = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": surgeon}], "remove_duplicate_initializers": False},
        disable_search=True,
    )
    output_model = graph_surgeries.run(ONNXModelHandler(model_path=str(model_path)), tmp_path / f"{name}_output")
    output = output_model.load_model()
    onnx.checker.check_model(output)
    return output


def count_ops(model):
    return Counter(node.op_type for node in ir.from_proto(model).graph)
