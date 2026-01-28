# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from onnxruntime import __version__ as ort_version
from packaging import version

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mnb_to_qdq import MatMulNBitsToQDQ
from olive.passes.onnx.onnx_dag import OnnxDAG

ORT_VERSION = version.parse(ort_version)
SKIP_2BIT = version.parse("1.24.0") > ORT_VERSION or version.parse(onnx.__version__) < version.parse("1.20.1")


@pytest.fixture(
    params=[
        pytest.param(
            (True, 2), marks=pytest.mark.skipif(SKIP_2BIT, reason="2-bit not supported in this version of ONNX Runtime")
        ),
        pytest.param(
            (False, 2),
            marks=pytest.mark.skipif(SKIP_2BIT, reason="2-bit not supported in this version of ONNX Runtime"),
        ),
        (True, 4),
        (False, 4),
        (True, 8),
        (False, 8),
    ],
    ids=["symmetric-2bit", "asymmetric-2bit", "symmetric-4bit", "asymmetric-4bit", "symmetric-8bit", "asymmetric-8bit"],
    name="create_mnb_model",
)
def create_mnb_model_fixture(request, tmp_path):
    symmetric, bits = request.param
    if version.parse("1.22.0") > ORT_VERSION:
        if bits == 8:
            pytest.skip("MatMulNBitsQuantizer doesn't support 8 bits in this version of ONNX Runtime")

        from onnxruntime.quantization.matmul_4bits_quantizer import (
            DefaultWeightOnlyQuantConfig,
        )
        from onnxruntime.quantization.matmul_4bits_quantizer import (
            MatMul4BitsQuantizer as MatMulNBitsQuantizer,
        )
    else:
        from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    block_size = 32
    # odd, %32 != 0
    in_dim = 33
    # even, %32 == 0
    hidden_dim = 64
    out_dim = 32

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = torch.nn.Linear(in_dim, hidden_dim, bias=True)
            self.f2 = torch.nn.Linear(hidden_dim, out_dim, bias=False)
            # equivalent to per-axis quantization
            self.f3 = torch.nn.Linear(out_dim, out_dim, bias=False)

        def forward(self, x):
            return self.f3(self.f2(self.f1(x)))

    model = TestModel()

    # base model
    base_path = tmp_path / "base.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 1, in_dim),
        base_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
        dynamo=True,
    )

    # quantized model
    mnb_path = tmp_path / "mnb.onnx"
    quant = MatMulNBitsQuantizer(
        onnx.load(base_path),
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=symmetric,
            bits=bits,
            # 8 bit only supports acc level 4 currently
            accuracy_level=4 if bits == 8 else 0,
        ),
    )
    quant.process()
    onnx.save(quant.model.model, mnb_path)

    return mnb_path, in_dim, symmetric, bits


@pytest.mark.parametrize("use_transpose_op", [True, False])
@pytest.mark.parametrize("use_signed_int", [True, False])
@pytest.mark.parametrize("add_zero_point", [True, False])
# Note: With dynamo export, the node names are like "node_MatMul_1_Q4" instead of "/f1/MatMul_Q4"
@pytest.mark.parametrize("nodes_to_exclude", [None, ["node_MatMul_1_Q4"]])
def test_mnb_to_qdq(create_mnb_model, nodes_to_exclude, add_zero_point, use_signed_int, use_transpose_op, tmp_path):
    mnb_path, in_dim, is_symmetric, bits = create_mnb_model

    if use_transpose_op and bits == 2:
        pytest.skip("Transpose op not yet supported for 2 bit in ONNX Runtime")

    input_model = ONNXModelHandler(mnb_path)

    # setup
    if nodes_to_exclude is not None:
        nodes_to_exclude = [name.replace("Q4", f"Q{bits}") for name in nodes_to_exclude]
    p = create_pass_from_dict(
        MatMulNBitsToQDQ,
        {
            "use_transpose_op": use_transpose_op,
            "use_signed_int": use_signed_int,
            "add_zero_point": add_zero_point,
            "nodes_to_exclude": nodes_to_exclude,
        },
        disable_search=True,
    )
    output_folder = tmp_path / "qdq-model"

    # execute
    qdq_model: ONNXModelHandler = p.run(input_model, output_folder)

    # count ops
    num_matmuls = 0
    num_mnbs = 0
    dag = OnnxDAG.from_model_path(qdq_model.model_path)
    for name in dag.get_node_names():
        op_type = dag.get_node_op_type(name)
        if op_type == "MatMul":
            num_matmuls += 1
        elif op_type == "MatMulNBits":
            num_mnbs += 1
    assert num_matmuls == 3 - len(nodes_to_exclude or [])
    assert num_mnbs == len(nodes_to_exclude or [])

    # validate
    original_session = onnxruntime.InferenceSession(str(mnb_path))
    original_session.disable_fallback()
    if is_symmetric and use_signed_int and not add_zero_point and use_transpose_op:
        # there seems to be a bug in ORT graph optimization which changes the int4 DQ to uint8 DQ
        with pytest.raises(Exception, match="uint8"):
            onnxruntime.InferenceSession(str(qdq_model.model_path))
        return
    else:
        qdq_session = onnxruntime.InferenceSession(str(qdq_model.model_path))
        qdq_session.disable_fallback()

    input_data = {"input": np.random.randn(1, 1, in_dim).astype(np.float32)}
    original_output = original_session.run(None, input_data)[0]
    qdq_output = qdq_session.run(None, input_data)[0]
    assert original_output.shape == qdq_output.shape
    assert original_output.dtype == qdq_output.dtype
    if bits == 4 and not use_transpose_op:
        # Pre transposed DQ model does not match the expected output on x64 CPU
        # check for assertion failure so we know when the test is fixed
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(original_output, qdq_output, atol=1e-4)
    else:
        # acc level 4 is used for 8 bit, so the tolerance is higher
        np.testing.assert_allclose(original_output, qdq_output, atol=1e-2 if bits == 8 else 1e-4)
