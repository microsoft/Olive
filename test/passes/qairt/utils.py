# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from pathlib import Path


def make_minimal_onnx(path: Path) -> None:
    """Write a minimal ONNX model to *path* for use in tests.

    Mirrors the LM_EXECUTOR shape: INT32 input_ids → Cast → FLOAT logits.
    """
    import onnx
    from onnx import TensorProto

    input_tensor = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])
    output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
    node = onnx.helper.make_node("Cast", inputs=["input_ids"], outputs=["cast_out"], to=TensorProto.FLOAT)
    reshape = onnx.helper.make_node("Unsqueeze", inputs=["cast_out", "axes"], outputs=["logits"])
    axes = onnx.helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    graph = onnx.helper.make_graph([node, reshape], "g", [input_tensor], [output_tensor], initializer=[axes])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
    onnx.save(model, str(path))
