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
@pytest.mark.parametrize("block_size", [16, 33], ids=["blockwise", "exact_k_per_channel"])
@pytest.mark.parametrize("bits", [4, 8])
def test_mnb_to_qdq_preserves_quantized_weights(tmp_path, bits, block_size, use_transpose_op):
    k, n = 33, 2
    num_blocks = (k + block_size - 1) // block_size
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1 << bits, (n, num_blocks * block_size), dtype=np.uint8)
    if bits == 4:
        storage = codes.reshape(n, num_blocks, block_size)
        if block_size % 2:
            storage = np.pad(storage, ((0, 0), (0, 0), (0, 1)))
        packed = storage[:, :, ::2] | (storage[:, :, 1::2] << 4)
    else:
        packed = codes.reshape(n, num_blocks, block_size)
    scales = np.arange(1, n * num_blocks + 1, dtype=np.float32).reshape(n, num_blocks) / 8
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "MatMulNBits",
                ["x", "weight", "scales"],
                ["y"],
                name="MatMulNBits",
                domain="com.microsoft",
                K=k,
                N=n,
                block_size=block_size,
                bits=bits,
            )
        ],
        "preserve_quantization",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, k])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, n])],
        initializer=[
            onnx.numpy_helper.from_array(packed, "weight"),
            onnx.numpy_helper.from_array(scales, "scales"),
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)]
    )
    path = tmp_path / "mnb.onnx"
    onnx.save(model, path)
    output = create_pass_from_dict(
        MatMulNBitsToQDQ,
        {"use_signed_int": True, "add_zero_point": True, "use_transpose_op": use_transpose_op},
        disable_search=True,
    ).run(ONNXModelHandler(path), tmp_path / "qdq")
    converted = onnx.load(output.model_path)
    onnx.checker.check_model(converted)
    initializers = {init.name: onnx.numpy_helper.to_array(init) for init in converted.graph.initializer}
    dq = next(node for node in converted.graph.node if node.op_type == "DequantizeLinear")
    weight_tensor = next(tensor for tensor in converted.graph.initializer if tensor.name == dq.input[0])
    assert weight_tensor.data_type == (onnx.TensorProto.INT4 if bits == 4 else onnx.TensorProto.INT8)
    converted_codes = initializers[dq.input[0]].astype(np.int16)
    converted_scales = initializers[dq.input[1]]
    zero_points = initializers[dq.input[2]].astype(np.int16)
    if not use_transpose_op:
        converted_codes = converted_codes.T
        converted_scales = converted_scales.T
    expected_codes = codes[:, :k].astype(np.int16) - (1 << (bits - 1))
    np.testing.assert_array_equal(converted_codes, expected_codes)
    np.testing.assert_array_equal(converted_scales, scales.flatten() if num_blocks == 1 else scales)
    np.testing.assert_array_equal(zero_points, np.zeros_like(zero_points))
    attrs = {item.name: onnx.helper.get_attribute_value(item) for item in dq.attribute}
    if num_blocks == 1:
        assert "block_size" not in attrs
        assert attrs["axis"] == (0 if use_transpose_op else 1)
    else:
        assert attrs["block_size"] == block_size
        assert attrs["axis"] == (1 if use_transpose_op else 0)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("zero_dtype", [np.float16, np.float32])
@pytest.mark.parametrize(("zero_point", "convert_other_node"), [(8.0, False), (8.5, True)])
def test_mnb_to_qdq_preserves_floating_point_zero_points(tmp_path, bits, zero_dtype, zero_point, convert_other_node):
    k, n = 32, 2
    packed = np.arange(n * k // (8 // bits), dtype=np.uint8).reshape(n, 1, -1)
    scales = np.array([[0.25], [0.5]], dtype=zero_dtype)
    zeros = np.array([[zero_point], [zero_point - 1]], dtype=zero_dtype)
    initializers = [
        onnx.numpy_helper.from_array(packed, "weight"),
        onnx.numpy_helper.from_array(scales, "scales"),
        onnx.numpy_helper.from_array(zeros, "float_zeros"),
    ]
    nodes = [
        onnx.helper.make_node(
            "MatMulNBits",
            ["x", "weight", "scales", "float_zeros"],
            ["y"],
            name="floating_zero_points",
            domain="com.microsoft",
            K=k,
            N=n,
            block_size=k,
            bits=bits,
        )
    ]
    float_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(zero_dtype))
    outputs = [onnx.helper.make_tensor_value_info("y", float_type, [1, n])]
    if convert_other_node:
        initializers.append(onnx.numpy_helper.from_array(np.full((n, 1), 1 << (bits - 1), np.uint8), "integer_zeros"))
        nodes.append(
            onnx.helper.make_node(
                "MatMulNBits",
                ["x", "weight", "scales", "integer_zeros"],
                ["converted_y"],
                name="integer_zero_points",
                domain="com.microsoft",
                K=k,
                N=n,
                block_size=k,
                bits=bits,
            )
        )
        outputs.append(onnx.helper.make_tensor_value_info("converted_y", float_type, [1, n]))
    graph = onnx.helper.make_graph(
        nodes,
        "floating_point_zero_points",
        [onnx.helper.make_tensor_value_info("x", float_type, [1, k])],
        outputs,
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)]
    )
    path = tmp_path / "mnb.onnx"
    onnx.save(model, path)
    output = create_pass_from_dict(
        MatMulNBitsToQDQ, {"use_signed_int": True, "add_zero_point": True}, disable_search=True
    ).run(ONNXModelHandler(path), tmp_path / "qdq")

    converted = onnx.load(output.model_path)
    onnx.checker.check_model(converted)
    remaining = [node for node in converted.graph.node if node.op_type == "MatMulNBits"]
    assert len(remaining) == 1
    assert remaining[0] == nodes[0]
    actual_initializers = {tensor.name: tensor for tensor in converted.graph.initializer}
    for original in initializers[:3]:
        actual = actual_initializers[original.name]
        assert actual.data_type == original.data_type
        np.testing.assert_array_equal(onnx.numpy_helper.to_array(actual), onnx.numpy_helper.to_array(original))
    assert sum(node.op_type == "DequantizeLinear" for node in converted.graph.node) == int(convert_other_node)
    assert sum(node.op_type == "MatMul" for node in converted.graph.node) == int(convert_other_node)
