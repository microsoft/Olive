# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from pathlib import Path


def make_minimal_onnx(path: Path) -> None:
    """Write a minimal ONNX Identity model to *path* for use in tests."""
    import onnx
    from onnx import TensorProto

    input_tensor = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "seq"])
    output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab"])
    node = onnx.helper.make_node("Identity", inputs=["input_ids"], outputs=["logits"])
    graph = onnx.helper.make_graph([node], "g", [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
    onnx.save(model, path)
