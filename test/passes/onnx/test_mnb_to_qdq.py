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

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TestModel()
        example_input = torch.randn(1, 1, in_dim)

    # base model
    base_path = tmp_path / "base.onnx"
    torch.onnx.export(
        model,
        example_input,
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
    qdq_model_proto = onnx.load(qdq_model.model_path)
    for node in qdq_model_proto.graph.node:
        op_type = node.op_type
        if op_type == "MatMul":
            num_matmuls += 1
        elif op_type == "MatMulNBits":
            num_mnbs += 1
    assert num_matmuls == 3 - len(nodes_to_exclude or [])
    assert num_mnbs == len(nodes_to_exclude or [])

    # validate
    original_session = onnxruntime.InferenceSession(str(mnb_path))
    original_session.disable_fallback()
    # disable qdq to mnb fusion so we can test the output of the DQ nodes directly
    disabled_optimizers = ["QDQSelectorActionTransformer"]
    try:
        qdq_session = onnxruntime.InferenceSession(str(qdq_model.model_path), disabled_optimizers=disabled_optimizers)
    except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
        if is_symmetric and use_signed_int and not add_zero_point and use_transpose_op and "uint8" in str(e):
            # Older ORT versions incorrectly require uint8 here. Newer versions accept the signed type.
            return
        raise
    qdq_session.disable_fallback()

    input_data = {"input": np.random.default_rng(0).standard_normal((1, 1, in_dim)).astype(np.float32)}
    original_output = original_session.run(None, input_data)[0]
    qdq_output = qdq_session.run(None, input_data)[0]
    assert original_output.shape == qdq_output.shape
    assert original_output.dtype == qdq_output.dtype
    # acc level 4 is used for 8 bit, so the tolerance is higher
    np.testing.assert_allclose(original_output, qdq_output, atol=2e-2 if bits == 8 else 1e-4)


@pytest.mark.parametrize("use_transpose_op", [True, False])
@pytest.mark.parametrize("add_zero_point", [True, False])
def test_mnb_to_qdq_int8_per_channel(create_mnb_model, add_zero_point, use_transpose_op, tmp_path):
    """Cover the int8 per-channel re-encoding.

    The fixture's first Linear has in_dim 33 against a block size of 32, so the final block is
    short. That is the case where broadcasting one scale per block over the weight columns has to
    repeat by the block size rather than by K / num_k_blocks.
    """
    mnb_path, in_dim, _, bits = create_mnb_model

    if use_transpose_op and bits == 2:
        pytest.skip("Transpose op not yet supported for 2 bit in ONNX Runtime")

    p = create_pass_from_dict(
        MatMulNBitsToQDQ,
        {
            "use_int8_per_channel": True,
            "use_transpose_op": use_transpose_op,
            "add_zero_point": add_zero_point,
        },
        disable_search=True,
    )
    qdq_model: ONNXModelHandler = p.run(ONNXModelHandler(mnb_path), tmp_path / "qdq-model")

    qdq_model_proto = onnx.load(qdq_model.model_path)
    initializers = {init.name: onnx.numpy_helper.to_array(init) for init in qdq_model_proto.graph.initializer}
    dq_nodes = [node for node in qdq_model_proto.graph.node if node.op_type == "DequantizeLinear"]

    assert len(dq_nodes) == 3
    assert not [node for node in qdq_model_proto.graph.node if node.op_type == "MatMulNBits"]

    for dq_node in dq_nodes:
        weight = initializers[dq_node.input[0]]
        scales = initializers[dq_node.input[1]]
        (axis,) = [attribute.i for attribute in dq_node.attribute if attribute.name == "axis"]

        # re-encoded as signed int8 regardless of the source bit width
        assert weight.dtype == np.int8
        # a per-channel DQ must not carry a block size
        assert not [attribute for attribute in dq_node.attribute if attribute.name == "block_size"]
        # the stored weight is N x K with the transpose op and K x N without it, so the axis the
        # scales apply along differs between the two modes
        assert axis == (0 if use_transpose_op else 1)
        assert scales.ndim == 1
        assert weight.shape[axis] == scales.shape[0]

        if add_zero_point:
            zero_points = initializers[dq_node.input[2]]
            # the re-encoding is symmetric, so the zero points exist only to satisfy consumers
            # that require the input and must not shift the weight
            assert zero_points.dtype == np.int8
            assert zero_points.shape == scales.shape
            assert not zero_points.any()
        else:
            assert len(dq_node.input) == 2

    # validate against the model the weights were re-encoded from
    original_session = onnxruntime.InferenceSession(str(mnb_path))
    original_session.disable_fallback()
    qdq_session = onnxruntime.InferenceSession(
        str(qdq_model.model_path), disabled_optimizers=["QDQSelectorActionTransformer"]
    )
    qdq_session.disable_fallback()

    input_data = {"input": np.random.default_rng(0).standard_normal((1, 1, in_dim)).astype(np.float32)}
    original_output = original_session.run(None, input_data)[0]
    qdq_output = qdq_session.run(None, input_data)[0]
    assert original_output.shape == qdq_output.shape
    assert original_output.dtype == qdq_output.dtype
    # this path requantizes an already quantized weight, so it is lossy by construction: one int8
    # scale per output channel replaces one scale per block. The bound is loose enough to absorb
    # that but tight enough to catch a wrong axis, a wrong transpose or a misaligned final block,
    # each of which scrambles the weight rather than perturbing it.
    np.testing.assert_allclose(original_output, qdq_output, rtol=0.05, atol=5e-2)


@pytest.mark.parametrize("zero_type", [onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT, onnx.TensorProto.BFLOAT16])
@pytest.mark.parametrize("use_transpose_op", [False, True])
def test_mnb_to_qdq_int8_per_channel_float_zero_points(tmp_path, zero_type, use_transpose_op):
    k, n, block_size = 33, 2, 32
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, (n, 2, block_size // 2), dtype=np.uint8)
    scales = np.array([[0.125, 0.25], [0.25, 0.125]], dtype=np.float32)
    zeros = np.array([[3.5, 8], [7, 11.5]], dtype=np.float32)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "MatMulNBits",
                ["x", "weight", "scales", "zeros"],
                ["y"],
                name="MatMulNBits",
                domain="com.microsoft",
                K=k,
                N=n,
                block_size=block_size,
                bits=4,
            )
        ],
        "floating_zeros",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, k])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, n])],
        initializer=[
            onnx.numpy_helper.from_array(packed, "weight"),
            onnx.numpy_helper.from_array(scales, "scales"),
            onnx.helper.make_tensor("zeros", zero_type, [n, 2], zeros.flatten().tolist()),
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)]
    )
    path = tmp_path / "floating_zeros.onnx"
    onnx.save(model, path)
    output = create_pass_from_dict(
        MatMulNBitsToQDQ,
        {"use_int8_per_channel": True, "use_transpose_op": use_transpose_op},
        disable_search=True,
    ).run(ONNXModelHandler(path), tmp_path / "qdq")
    converted = onnx.load(output.model_path)
    onnx.checker.check_model(converted)
    initializers = {init.name: onnx.numpy_helper.to_array(init) for init in converted.graph.initializer}
    dq = next(node for node in converted.graph.node if node.op_type == "DequantizeLinear")
    quantized = initializers[dq.input[0]]
    if not use_transpose_op:
        quantized = quantized.T
    converted_scales = initializers[dq.input[1]]
    unpacked = np.stack((packed & 15, packed >> 4), axis=-1).reshape(n, -1)[:, :k]
    expected = (unpacked - np.repeat(zeros, block_size, axis=1)[:, :k]) * np.repeat(scales, block_size, axis=1)[:, :k]
    np.testing.assert_allclose(converted_scales, np.abs(expected).max(axis=1) / 127)
    assert np.max(np.abs(quantized * converted_scales[:, None] - expected)) <= converted_scales.max() / 2 + 1e-6
    if zero_type == onnx.TensorProto.FLOAT:
        inputs = {"x": rng.standard_normal((1, k)).astype(np.float32)}
        original = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        actual = onnxruntime.InferenceSession(output.model_path, providers=["CPUExecutionProvider"]).run(None, inputs)[
            0
        ]
        np.testing.assert_allclose(actual, original, rtol=0.05, atol=0.1)
