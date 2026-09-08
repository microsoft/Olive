# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from collections import Counter

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries
from olive.passes.onnx.graph_surgery import Surgeon
from olive.passes.onnx.graph_surgery import lowering as lowering_surgeries


def _apply_surgery(tmp_path, model: ir.Model, surgeon: str, *, case: str = "model") -> ir.Model:
    model_path = tmp_path / f"{case}.onnx"
    ir.save(model, model_path)
    olive_model = ONNXModelHandler(model_path=str(model_path))
    graph_surgeries = create_pass_from_dict(
        GraphSurgeries,
        {
            "surgeries": [{"surgeon": surgeon}],
            "remove_duplicate_initializers": False,
        },
        disable_search=True,
    )
    output_model = graph_surgeries.run(olive_model, str(tmp_path / f"{case}_output"))
    return output_model.load_ir_model()


def _run(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, feeds)


def _counts(model: ir.Model) -> Counter:
    return Counter(node.op_type for node in model.graph.all_nodes())


def _metadata(value: ir.Value):
    return value.name, value.dtype, tuple(value.shape) if value.shape is not None else None


def test_lowering_module_registers_all_surgeons():
    expected = {
        lowering_surgeries.ClipToMinMax,
        lowering_surgeries.Rank4RMSNormToRank3,
        lowering_surgeries.DecomposeOnnxRotaryEmbedding,
        lowering_surgeries.TensorScatterToScatterND,
        lowering_surgeries.DecomposeAttention,
        lowering_surgeries.StaticEmptyKV,
    }
    assert {Surgeon.registry[surgeon.__name__.lower()] for surgeon in expected} == expected


def _clip_model(dtype: ir.DataType, *, both_bounds: bool = True) -> ir.Model:
    numpy_dtype = ml_dtypes.bfloat16 if dtype == ir.DataType.BFLOAT16 else np.float32
    x = ir.Value(name="x", type=ir.TensorType(dtype), shape=ir.Shape(["batch", 4]))
    lower = ir.Value(
        name="lower",
        type=ir.TensorType(dtype),
        const_value=ir.tensor(np.array(-1.0, dtype=numpy_dtype)),
    )
    inputs = [x, lower]
    initializers = [lower]
    if both_bounds:
        upper = ir.Value(
            name="upper",
            type=ir.TensorType(dtype),
            const_value=ir.tensor(np.array(1.0, dtype=numpy_dtype)),
        )
        inputs.append(upper)
        initializers.append(upper)
    y = ir.Value(name="y", type=ir.TensorType(dtype), shape=ir.Shape(["batch", 4]))
    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[ir.Node("", "Clip", inputs=inputs, outputs=[y], name="clip")],
        initializers=initializers,
        opset_imports={"": 24},
        name="clip",
    )
    return ir.Model(graph, ir_version=10)


@pytest.mark.parametrize(
    ("both_bounds", "expected"),
    [(True, Counter({"Max": 1, "Min": 1})), (False, Counter({"Max": 1}))],
)
def test_clip_to_min_max_lowers_bfloat16_and_preserves_metadata(tmp_path, both_bounds, expected):
    model = _clip_model(ir.DataType.BFLOAT16, both_bounds=both_bounds)
    output_metadata = _metadata(model.graph.outputs[0])

    lowered = _apply_surgery(tmp_path, model, "ClipToMinMax")

    assert _counts(lowered) == expected
    assert _metadata(lowered.graph.outputs[0]) == output_metadata


def test_clip_to_min_max_leaves_other_dtypes_unchanged(tmp_path):
    lowered = _apply_surgery(tmp_path, _clip_model(ir.DataType.FLOAT), "ClipToMinMax")
    assert _counts(lowered) == Counter({"Clip": 1})


def _rmsnorm_model(shape, *, axis=-1, epsilon=1e-6) -> ir.Model:
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
    weight = ir.Value(
        name="weight",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([shape[-1]]),
        const_value=ir.tensor(np.linspace(0.5, 1.5, shape[-1], dtype=np.float32)),
    )
    attributes = {"axis": axis}
    if epsilon is not None:
        attributes["epsilon"] = epsilon
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
    node = ir.Node(
        "",
        "RMSNormalization",
        inputs=[x, weight],
        outputs=[y],
        attributes=ir.convenience.convert_attributes(attributes),
    )
    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[node],
        initializers=[weight],
        opset_imports={"": 24},
        name="rmsnorm",
    )
    return ir.Model(graph, ir_version=10)


def test_rank4_rmsnorm_to_rank3_is_exact_and_preserves_metadata(tmp_path):
    model = _rmsnorm_model([1, 3, 4, 8])
    feeds = {"x": np.random.default_rng(0).standard_normal((1, 3, 4, 8)).astype(np.float32)}
    reference = _run(model, feeds)[0]
    output_metadata = _metadata(model.graph.outputs[0])

    lowered = _apply_surgery(tmp_path, model, "Rank4RMSNormToRank3")

    assert _counts(lowered)["RMSNormalization"] == 1
    assert _counts(lowered)["Reshape"] == 2
    assert _metadata(lowered.graph.outputs[0]) == output_metadata
    np.testing.assert_allclose(_run(lowered, feeds)[0], reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("shape", "axis", "epsilon"),
    [
        ([1, 3, 8], -1, 1e-6),
        ([1, 3, "heads", 8], -1, 1e-6),
        ([1, 3, 4, 8], 2, 1e-6),
        ([1, 3, 4, 8], -1, None),
    ],
)
def test_rank4_rmsnorm_to_rank3_rejects_unsupported_forms(tmp_path, shape, axis, epsilon):
    lowered = _apply_surgery(
        tmp_path,
        _rmsnorm_model(shape, axis=axis, epsilon=epsilon),
        "Rank4RMSNormToRank3",
    )
    counts = _counts(lowered)
    assert counts["RMSNormalization"] == 1
    assert counts.get("Reshape", 0) == 0


def _rope_model(
    batch=1,
    sequence=4,
    num_heads=2,
    head_dimension=8,
    *,
    domain="",
    interleaved=0,
    rotary_embedding_dim=0,
    cos_rank=3,
) -> ir.Model:
    hidden_size = num_heads * head_dimension
    half_dimension = head_dimension // 2
    x = ir.Value(
        name="x",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([batch, sequence, hidden_size]),
    )
    if domain:
        position_ids = ir.Value(
            name="position_ids",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([batch, sequence]),
        )
        cos = ir.Value(
            name="cos",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([32, head_dimension]),
        )
        sin = ir.Value(
            name="sin",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([32, head_dimension]),
        )
        inputs = [x, position_ids, cos, sin]
        graph_inputs = inputs
    else:
        cos_shape = [batch, sequence, half_dimension] if cos_rank == 3 else [sequence, half_dimension]
        cos = ir.Value(name="cos", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(cos_shape))
        sin = ir.Value(name="sin", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(cos_shape))
        inputs = [x, cos, sin]
        graph_inputs = inputs

    y = ir.Value(
        name="y",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([batch, sequence, hidden_size]),
    )
    node = ir.Node(
        domain,
        "RotaryEmbedding",
        inputs=inputs,
        outputs=[y],
        attributes=ir.convenience.convert_attributes(
            {
                "num_heads": num_heads,
                "rotary_embedding_dim": rotary_embedding_dim,
                "interleaved": interleaved,
            }
        ),
    )
    opsets = {"": 24}
    if domain:
        opsets[domain] = 1
    graph = ir.Graph(
        inputs=graph_inputs,
        outputs=[y],
        nodes=[node],
        opset_imports=opsets,
        name="rope",
    )
    return ir.Model(graph, ir_version=10)


def _rope_feeds(batch=1, sequence=4, num_heads=2, head_dimension=8):
    rng = np.random.default_rng(1)
    return {
        "x": rng.standard_normal((batch, sequence, num_heads * head_dimension)).astype(np.float32),
        "cos": rng.standard_normal((batch, sequence, head_dimension // 2)).astype(np.float32),
        "sin": rng.standard_normal((batch, sequence, head_dimension // 2)).astype(np.float32),
    }


def test_decompose_onnx_rotary_embedding_is_exact_and_preserves_metadata(tmp_path):
    model = _rope_model(batch=2, sequence=3, num_heads=4, head_dimension=16)
    feeds = _rope_feeds(batch=2, sequence=3, num_heads=4, head_dimension=16)
    reference = _run(model, feeds)[0]
    output_metadata = _metadata(model.graph.outputs[0])

    lowered = _apply_surgery(tmp_path, model, "DecomposeOnnxRotaryEmbedding")

    counts = _counts(lowered)
    assert counts.get("RotaryEmbedding", 0) == 0
    assert counts["Slice"] == 2
    assert counts["Concat"] == 1
    assert _metadata(lowered.graph.outputs[0]) == output_metadata
    np.testing.assert_allclose(_run(lowered, feeds)[0], reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("interleaved", "rotary_embedding_dim", "cos_rank"),
    [(1, 0, 3), (0, 4, 3), (0, 0, 2)],
)
def test_decompose_onnx_rotary_embedding_rejects_unsupported_forms(
    tmp_path, interleaved, rotary_embedding_dim, cos_rank
):
    model = _rope_model(
        interleaved=interleaved,
        rotary_embedding_dim=rotary_embedding_dim,
        cos_rank=cos_rank,
    )
    lowered = _apply_surgery(tmp_path, model, "DecomposeOnnxRotaryEmbedding")
    assert _counts(lowered)["RotaryEmbedding"] == 1


def test_standard_and_microsoft_rotary_embedding_surgeries_are_distinct(tmp_path):
    microsoft_model = _rope_model(domain="com.microsoft")
    lowered_microsoft = _apply_surgery(
        tmp_path,
        microsoft_model,
        "DecomposeOnnxRotaryEmbedding",
        case="microsoft",
    )
    assert _counts(lowered_microsoft)["RotaryEmbedding"] == 1

    standard_model = _rope_model()
    lowered_standard = _apply_surgery(
        tmp_path,
        standard_model,
        "DecomposeRotaryEmbedding",
        case="standard",
    )
    assert _counts(lowered_standard)["RotaryEmbedding"] == 1


def _tensor_scatter_model(max_length=16, dimension=4, *, batch=1, axis=1, symbolic_length=False):
    cache_length = "max_length" if symbolic_length else max_length
    cache = ir.Value(
        name="cache",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([batch, cache_length, dimension]),
    )
    update = ir.Value(
        name="update",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([batch, "sequence", dimension]),
    )
    write_indices = ir.Value(
        name="write_indices",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape([1]),
    )
    y = ir.Value(
        name="y",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([batch, cache_length, dimension]),
    )
    node = ir.Node(
        "",
        "TensorScatter",
        inputs=[cache, update, write_indices],
        outputs=[y],
        attributes=ir.convenience.convert_attributes({"axis": axis}),
    )
    graph = ir.Graph(
        inputs=[cache, update, write_indices],
        outputs=[y],
        nodes=[node],
        opset_imports={"": 24},
        name="tensor_scatter",
    )
    return ir.Model(graph, ir_version=10)


def _tensor_scatter_feeds(max_length, dimension, sequence, start):
    rng = np.random.default_rng(sequence * 10 + start)
    return {
        "cache": rng.standard_normal((1, max_length, dimension)).astype(np.float32),
        "update": rng.standard_normal((1, sequence, dimension)).astype(np.float32),
        "write_indices": np.array([start], dtype=np.int64),
    }


@pytest.mark.parametrize(("sequence", "start"), [(5, 0), (1, 7)])
def test_tensor_scatter_to_scatternd_is_exact_and_preserves_metadata(tmp_path, sequence, start):
    model = _tensor_scatter_model()
    feeds = _tensor_scatter_feeds(16, 4, sequence, start)
    reference = _run(model, feeds)[0]
    output_metadata = _metadata(model.graph.outputs[0])

    lowered = _apply_surgery(tmp_path, model, "TensorScatterToScatterND")

    counts = _counts(lowered)
    assert counts.get("TensorScatter", 0) == 0
    assert counts["ScatterND"] == 1
    assert counts.get("Range", 0) == 0
    assert _metadata(lowered.graph.outputs[0]) == output_metadata
    np.testing.assert_array_equal(_run(lowered, feeds)[0], reference)


@pytest.mark.parametrize(
    ("batch", "axis", "symbolic_length"),
    [(2, 1, False), (1, 0, False), (1, 1, True)],
)
def test_tensor_scatter_to_scatternd_rejects_unsupported_forms(tmp_path, batch, axis, symbolic_length):
    model = _tensor_scatter_model(batch=batch, axis=axis, symbolic_length=symbolic_length)
    lowered = _apply_surgery(tmp_path, model, "TensorScatterToScatterND")
    assert _counts(lowered)["TensorScatter"] == 1
    assert _counts(lowered).get("ScatterND", 0) == 0


def _attention_model(
    *,
    batch=1,
    sequence=4,
    past_sequence=0,
    query_heads=8,
    kv_heads=2,
    head_dimension=16,
    with_mask=True,
    with_past=False,
    with_nonpadding=False,
    softcap=0.0,
    is_causal=1,
    rank=3,
    qk_matmul_output_mode=0,
    output_count=3,
) -> ir.Model:
    query_hidden = query_heads * head_dimension
    kv_hidden = kv_heads * head_dimension
    if rank == 3:
        query_shape = [batch, sequence, query_hidden]
        kv_shape = [batch, sequence, kv_hidden]
    else:
        query_shape = [batch, query_heads, sequence, head_dimension]
        kv_shape = [batch, kv_heads, sequence, head_dimension]
    query = ir.Value(name="query", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(query_shape))
    key = ir.Value(name="key", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(kv_shape))
    value = ir.Value(name="value", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(kv_shape))
    inputs = [query, key, value]
    graph_inputs = [query, key, value]

    if with_mask:
        mask = ir.Value(
            name="mask",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, 1, sequence, past_sequence + sequence]),
        )
        inputs.append(mask)
        graph_inputs.append(mask)
    else:
        inputs.append(None)
    if with_past:
        past_key = ir.Value(
            name="past_key",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, kv_heads, past_sequence, head_dimension]),
        )
        past_value = ir.Value(
            name="past_value",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, kv_heads, past_sequence, head_dimension]),
        )
        inputs.extend([past_key, past_value])
        graph_inputs.extend([past_key, past_value])
    elif with_nonpadding:
        inputs.extend([None, None])
    if with_nonpadding:
        nonpadding = ir.Value(
            name="nonpadding",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([batch]),
        )
        inputs.append(nonpadding)
        graph_inputs.append(nonpadding)

    outputs = [
        ir.Value(
            name="y",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, sequence, query_hidden]),
        ),
        ir.Value(
            name="present_key",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, kv_heads, past_sequence + sequence, head_dimension]),
        ),
        ir.Value(
            name="present_value",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, kv_heads, past_sequence + sequence, head_dimension]),
        ),
    ][:output_count]
    node = ir.Node(
        "",
        "Attention",
        inputs=inputs,
        outputs=outputs,
        attributes=ir.convenience.convert_attributes(
            {
                "q_num_heads": query_heads,
                "kv_num_heads": kv_heads,
                "scale": 1.0,
                "softcap": softcap,
                "is_causal": is_causal,
                "qk_matmul_output_mode": qk_matmul_output_mode,
            }
        ),
    )
    graph = ir.Graph(
        inputs=graph_inputs,
        outputs=outputs,
        nodes=[node],
        opset_imports={"": 24},
        name="attention",
    )
    return ir.Model(graph, ir_version=10)


def _attention_feeds(
    *,
    batch=1,
    sequence=4,
    past_sequence=0,
    query_heads=8,
    kv_heads=2,
    head_dimension=16,
    with_mask=True,
    with_past=False,
    with_nonpadding=False,
    **_,
):
    rng = np.random.default_rng(batch + sequence + past_sequence + kv_heads)
    feeds = {
        "query": rng.standard_normal((batch, sequence, query_heads * head_dimension)).astype(np.float32),
        "key": rng.standard_normal((batch, sequence, kv_heads * head_dimension)).astype(np.float32),
        "value": rng.standard_normal((batch, sequence, kv_heads * head_dimension)).astype(np.float32),
    }
    if with_mask:
        feeds["mask"] = (rng.standard_normal((batch, 1, sequence, past_sequence + sequence)) * 0.1).astype(np.float32)
    if with_past:
        feeds["past_key"] = rng.standard_normal((batch, kv_heads, past_sequence, head_dimension)).astype(np.float32)
        feeds["past_value"] = rng.standard_normal((batch, kv_heads, past_sequence, head_dimension)).astype(np.float32)
    if with_nonpadding:
        feeds["nonpadding"] = np.full((batch,), past_sequence + sequence - 1, dtype=np.int64)
    return feeds


_ATTENTION_CASES = [
    pytest.param(
        {
            "batch": 1,
            "sequence": 4,
            "query_heads": 8,
            "kv_heads": 8,
            "with_mask": True,
            "is_causal": 1,
        },
        0.0,
        id="prefill-mha",
    ),
    pytest.param(
        {
            "batch": 2,
            "sequence": 3,
            "query_heads": 8,
            "kv_heads": 2,
            "with_mask": False,
            "is_causal": 0,
        },
        0.0,
        id="batch2-gqa",
    ),
    pytest.param(
        {
            "batch": 1,
            "sequence": 1,
            "past_sequence": 3,
            "query_heads": 8,
            "kv_heads": 1,
            "with_mask": True,
            "with_past": True,
            "softcap": 30.0,
            "is_causal": 1,
        },
        1e-5,
        id="decode-gqa-past-softcap",
    ),
]


@pytest.mark.parametrize(("configuration", "atol"), _ATTENTION_CASES)
def test_decompose_attention_is_exact_and_preserves_outputs(tmp_path, configuration, atol):
    model = _attention_model(**configuration)
    feeds = _attention_feeds(**configuration)
    reference = _run(model, feeds)
    output_metadata = [_metadata(output) for output in model.graph.outputs]

    lowered = _apply_surgery(tmp_path, model, "DecomposeAttention")

    assert _counts(lowered).get("Attention", 0) == 0
    assert _counts(lowered)["Softmax"] == 1
    assert [_metadata(output) for output in lowered.graph.outputs] == output_metadata
    actual = _run(lowered, feeds)
    for expected, output in zip(reference, actual):
        np.testing.assert_allclose(output, expected, rtol=0, atol=atol)


def test_decompose_attention_nonpadding_uses_static_indices(tmp_path):
    configuration = {
        "with_mask": False,
        "with_nonpadding": True,
        "is_causal": 0,
    }
    model = _attention_model(**configuration)
    feeds = _attention_feeds(**configuration)
    reference = _run(model, feeds)

    lowered = _apply_surgery(tmp_path, model, "DecomposeAttention")

    counts = _counts(lowered)
    assert counts.get("Attention", 0) == 0
    assert counts.get("Range", 0) == 0
    assert counts["Less"] >= 1
    np.testing.assert_allclose(_run(lowered, feeds)[0], reference[0], rtol=0, atol=0)


def test_decompose_attention_rewires_single_output(tmp_path):
    model = _attention_model(output_count=1, with_mask=False, is_causal=0)
    feeds = _attention_feeds(with_mask=False)
    reference = _run(model, feeds)[0]

    lowered = _apply_surgery(tmp_path, model, "DecomposeAttention")

    assert len(lowered.graph.outputs) == 1
    assert _metadata(lowered.graph.outputs[0]) == _metadata(model.graph.outputs[0])
    np.testing.assert_allclose(_run(lowered, feeds)[0], reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("rank", "qk_matmul_output_mode"),
    [(4, 0), (3, 1)],
)
def test_decompose_attention_rejects_unsupported_forms(tmp_path, rank, qk_matmul_output_mode):
    model = _attention_model(
        rank=rank,
        qk_matmul_output_mode=qk_matmul_output_mode,
        output_count=1,
    )
    lowered = _apply_surgery(tmp_path, model, "DecomposeAttention")
    assert _counts(lowered)["Attention"] == 1


def _empty_kv_model(
    kv_hidden=128,
    *,
    dtype=ir.DataType.FLOAT16,
    cast_like=True,
    shape_start=0,
) -> ir.Model:
    query = ir.Value(
        name="query",
        type=ir.TensorType(dtype),
        shape=ir.Shape(["batch", "sequence", 64]),
    )
    builder = ir.tape.Tape()
    batch_dimension = builder.op(
        "Shape",
        [query],
        {"start": shape_start, "end": shape_start + 1},
    )
    shape_tail = builder.op("Constant", [], {"value_ints": [0, kv_hidden]})
    empty_shape = builder.op("Concat", [batch_dimension, shape_tail], {"axis": 0})
    empty = builder.op("ConstantOfShape", [empty_shape])
    output = builder.op("CastLike", [empty, query]) if cast_like else builder.op("Cast", [empty], {"to": int(dtype)})
    output.name = "empty_kv"
    output.dtype = dtype
    output.shape = ir.Shape([1, 0, kv_hidden])
    graph = ir.Graph(
        inputs=[query],
        outputs=[output],
        nodes=builder.nodes,
        opset_imports={"": 24},
        name="empty_kv",
    )
    return ir.Model(graph, ir_version=10)


@pytest.mark.parametrize(
    ("cast_like", "dtype"),
    [
        (True, ir.DataType.FLOAT16),
        (False, ir.DataType.FLOAT16),
        (False, ir.DataType.BFLOAT16),
    ],
)
def test_static_empty_kv_preserves_shape_dtype_and_output_metadata(tmp_path, cast_like, dtype):
    model = _empty_kv_model(cast_like=cast_like, dtype=dtype)
    output_metadata = _metadata(model.graph.outputs[0])

    lowered = _apply_surgery(tmp_path, model, "StaticEmptyKV")

    counts = _counts(lowered)
    assert counts.get("Shape", 0) == 0
    assert counts.get("ConstantOfShape", 0) == 0
    assert counts.get("CastLike", 0) == 0
    assert counts.get("Cast", 0) == 0
    assert _metadata(lowered.graph.outputs[0]) == output_metadata
    constant = lowered.graph.outputs[0].producer().attributes["value"].value
    assert tuple(constant.shape) == (1, 0, 128)
    assert constant.dtype == dtype


def test_static_empty_kv_rejects_nonbatch_shape(tmp_path):
    model = _empty_kv_model(shape_start=1)
    lowered = _apply_surgery(tmp_path, model, "StaticEmptyKV")
    counts = _counts(lowered)
    assert counts["Shape"] == 1
    assert counts["ConstantOfShape"] == 1
    assert counts["CastLike"] == 1
