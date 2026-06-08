# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

from pathlib import Path


def make_minimal_onnx(path: Path) -> None:
    """Write a minimal ONNX model to *path* for use in tests.

    Mirrors the LM_EXECUTOR shape: INT32 input_ids [batch, seq] →
    Cast → FLOAT → Reshape → FLOAT logits [batch, 1, vocab_size].
    The graph is schema-valid for opset 14 and passes onnx.checker.
    """
    import onnx
    from onnx import TensorProto

    input_tensor = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])
    output_tensor = onnx.helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 1, "vocab_size"])
    cast_node = onnx.helper.make_node("Cast", inputs=["input_ids"], outputs=["cast_out"], to=TensorProto.FLOAT)
    reshape_node = onnx.helper.make_node("Reshape", inputs=["cast_out", "new_shape"], outputs=["logits"])
    new_shape = onnx.helper.make_tensor("new_shape", TensorProto.INT64, [3], [0, 1, -1])
    graph = onnx.helper.make_graph(
        [cast_node, reshape_node], "g", [input_tensor], [output_tensor], initializer=[new_shape]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
