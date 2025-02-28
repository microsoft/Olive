# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from packaging import version

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mnb_to_qdq import MatMulNBitsToQDQ
from olive.passes.onnx.onnx_dag import OnnxDAG


@pytest.fixture(params=[True, False], ids=["symmetric", "asymmetric"], name="create_mnb_model")
def create_mnb_model_fixture(request, tmp_path):
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

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
    )

    # quantized model
    mnb_path = tmp_path / "mnb.onnx"
    quant = MatMul4BitsQuantizer(onnx.load(base_path), block_size=block_size, is_symmetric=request.param)
    quant.process()
    onnx.save(quant.model.model, mnb_path)

    return mnb_path, in_dim, request.param


@pytest.mark.skipif(
    version.parse(onnxruntime.__version__) < version.parse("1.20"),
    reason="Int4 DQ is only supported in ORT >= 1.20",
)
@pytest.mark.parametrize("use_transpose_op", [True, False])
@pytest.mark.parametrize("use_int4", [True, False])
@pytest.mark.parametrize("add_zero_point", [True, False])
@pytest.mark.parametrize("nodes_to_exclude", [None, ["/f1/MatMul_Q4"]])
@pytest.mark.parametrize("execution_provider", ["CPUExecutionProvider"])
def test_mnb_to_qdq(
    create_mnb_model, execution_provider, nodes_to_exclude, add_zero_point, use_int4, use_transpose_op, tmp_path
):
    available_providers = onnxruntime.get_available_providers()
    if execution_provider not in available_providers:
        pytest.skip(f"{execution_provider} is not available on this system {available_providers}")

    mnb_path, in_dim, is_symmetric = create_mnb_model
    input_model = ONNXModelHandler(mnb_path)

    # setup
    p = create_pass_from_dict(
        MatMulNBitsToQDQ,
        {
            "use_transpose_op": use_transpose_op,
            "use_int4": use_int4,
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
    original_session = onnxruntime.InferenceSession(str(mnb_path), providers=[execution_provider])
    original_session.disable_fallback()
    if is_symmetric and use_int4 and not add_zero_point and use_transpose_op:
        # there seems to be a bug in ORT graph optimization which changes the int4 DQ to uint8 DQ
        with pytest.raises(Exception, match="uint8"):
            onnxruntime.InferenceSession(str(qdq_model.model_path), providers=[execution_provider])
        return
    else:
        qdq_session = onnxruntime.InferenceSession(str(qdq_model.model_path), providers=[execution_provider])
        qdq_session.disable_fallback()

    input_data = {"input": np.random.randn(1, 1, in_dim).astype(np.float32)}
    original_output = original_session.run(None, input_data)[0]
    qdq_output = qdq_session.run(None, input_data)[0]
    assert original_output.shape == qdq_output.shape
    assert original_output.dtype == qdq_output.dtype
    if execution_provider == "CPUExecutionProvider" and not use_transpose_op:
        # Pre transposed DQ model does not match the expected output on x64 CPU
        # check for assertion failure so we know when the test is fixed
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(original_output, qdq_output, rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_allclose(original_output, qdq_output, rtol=1e-4, atol=1e-4)
