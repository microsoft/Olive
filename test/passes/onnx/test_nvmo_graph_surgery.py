# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.nvmo_graph_surgery import NVModelOptGraphSurgery

pytest.importorskip("modelopt", reason="nvidia-modelopt required for graph surgery tests")
pytest.importorskip("transformers", reason="transformers required for GQA surgery tests")

MODEL_ID = "Qwen/Qwen2.5-0.5B"
VOCAB_SIZE = 64

_RNG = np.random.RandomState(42)


def _fp16(*shape):
    return (_RNG.randn(*shape) * 0.02).astype(np.float16)


def _init(name, arr):
    return numpy_helper.from_array(arr, name=name)


def _get_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", heads)
    hd = hidden // heads
    return hidden, heads, kv, hd


# ---------------------------------------------------------------------------
# Dummy model builders
# ---------------------------------------------------------------------------


def _build_attention_model(hidden_size, num_heads, kv_heads, head_dim) -> onnx.ModelProto:
    """Build a single-layer LLM with realistic Optimum-style attention.

    Simplified from Model-Optimizer test_gqa_graph_surgery._build_toy_model:
      Gather -> LayerNorm -> Q/K/V MatMul -> Reshape -> Transpose
      -> KV cache concat -> (GQA repeat) -> K^T -> scale*Q@K^T
      -> Add(attn_bias) -> Softmax -> Attn@V -> Transpose -> Reshape
      -> o_proj MatMul -> residual Add -> lm_head -> logits

    Attention mask is wired into a causal+padding bias subgraph.
    """
    q_dim = num_heads * head_dim
    k_dim = kv_heads * head_dim
    ap = "/model/layers.0/self_attn"

    nodes, inits = [], []

    # --- Constants ---
    inits.extend(
        [
            _init("one_f16", np.array(1.0, dtype=np.float16)),
            _init("neg_large_f16", np.array(-1e4, dtype=np.float16)),
            _init("axes_0", np.array([0], dtype=np.int64)),
            _init("axes_01", np.array([0, 1], dtype=np.int64)),
            _init("axes_12", np.array([1, 2], dtype=np.int64)),
            _init("trilu_k1", np.array(1, dtype=np.int64)),
        ]
    )

    # --- Graph I/O ---
    graph_inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "S"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "T"]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT16, ["B", kv_heads, "P", head_dim]),
        helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT16, ["B", kv_heads, "P", head_dim]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["B", "S", VOCAB_SIZE]),
        helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT16, ["B", kv_heads, "T", head_dim]),
        helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT16, ["B", kv_heads, "T", head_dim]),
    ]

    # --- Embedding ---
    inits.append(_init("model.embed_tokens.weight", _fp16(VOCAB_SIZE, hidden_size)))
    nodes.append(
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["/model/embed_tokens/Gather_output_0"],
            name="/model/embed_tokens/Gather",
            axis=0,
        )
    )
    hidden = "/model/embed_tokens/Gather_output_0"

    # --- Causal + padding mask (from attention_mask) ---
    nodes.append(helper.make_node("Shape", ["input_ids"], ["ids_shape"], name="/model/pos/Shape"))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/model/pos/C1_output_0"],
            name="/model/pos/C1",
            value=numpy_helper.from_array(np.array(1, dtype=np.int64), name=""),
        )
    )
    nodes.append(
        helper.make_node(
            "Gather",
            ["ids_shape", "/model/pos/C1_output_0"],
            ["seq_len"],
            name="/model/pos/seq_gather",
            axis=0,
        )
    )
    nodes.append(helper.make_node("Unsqueeze", ["seq_len", "axes_0"], ["seq_1d"], name="/model/causal/unsq"))
    nodes.append(helper.make_node("Concat", ["seq_1d", "seq_1d"], ["causal_shape"], name="/model/causal/cat", axis=0))
    nodes.append(
        helper.make_node(
            "ConstantOfShape",
            ["causal_shape"],
            ["causal_ones"],
            name="/model/causal/fill",
            value=numpy_helper.from_array(np.array([1.0], dtype=np.float16), name=""),
        )
    )
    nodes.append(
        helper.make_node("Trilu", ["causal_ones", "trilu_k1"], ["upper_tri"], name="/model/causal/trilu", upper=1)
    )
    nodes.append(helper.make_node("Mul", ["upper_tri", "neg_large_f16"], ["causal_4d_raw"], name="/model/causal/mul"))
    nodes.append(
        helper.make_node("Unsqueeze", ["causal_4d_raw", "axes_01"], ["causal_4d"], name="/model/causal/unsq4d")
    )
    nodes.append(helper.make_node("Cast", ["attention_mask"], ["pad_f16"], name="/model/pad/cast", to=10))
    nodes.append(helper.make_node("Unsqueeze", ["pad_f16", "axes_12"], ["pad_4d"], name="/model/pad/unsq"))
    nodes.append(helper.make_node("Sub", ["one_f16", "pad_4d"], ["inv_pad"], name="/model/pad/inv"))
    nodes.append(helper.make_node("Mul", ["inv_pad", "neg_large_f16"], ["pad_bias"], name="/model/pad/mul"))
    nodes.append(helper.make_node("Add", ["causal_4d", "pad_bias"], ["attn_bias"], name="/model/bias/add"))

    # --- LayerNorm ---
    ln_w = "model.layers.0.input_layernorm.weight"
    ln_b = "model.layers.0.input_layernorm.bias"
    inits.append(_init(ln_w, np.ones(hidden_size, dtype=np.float16)))
    inits.append(_init(ln_b, np.zeros(hidden_size, dtype=np.float16)))
    ln_out = "/model/layers.0/input_layernorm/Mul_1_output_0"
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [hidden, ln_w, ln_b],
            [ln_out],
            name="/model/layers.0/input_layernorm/LayerNorm",
            axis=-1,
            epsilon=1e-5,
        )
    )

    # --- Q / K / V projections ---
    inits.extend(
        [
            _init("model.layers.0.self_attn.q_proj.weight", _fp16(hidden_size, q_dim)),
            _init("model.layers.0.self_attn.k_proj.weight", _fp16(hidden_size, k_dim)),
            _init("model.layers.0.self_attn.v_proj.weight", _fp16(hidden_size, k_dim)),
            _init("model.layers.0.self_attn.o_proj.weight", _fp16(q_dim, hidden_size)),
        ]
    )
    for proj, _dim in [("q_proj", q_dim), ("k_proj", k_dim), ("v_proj", k_dim)]:
        nodes.append(
            helper.make_node(
                "MatMul",
                [ln_out, f"model.layers.0.self_attn.{proj}.weight"],
                [f"{ap}/{proj}/MatMul_output_0"],
                name=f"{ap}/{proj}/MatMul",
            )
        )

    # --- Reshape + Transpose to multi-head ---
    inits.append(_init(f"{ap}/q_shape", np.array([0, 0, num_heads, head_dim], np.int64)))
    inits.append(_init(f"{ap}/kv_shape", np.array([0, 0, kv_heads, head_dim], np.int64)))
    for tag, proj, shape_name in [
        ("", "q_proj", "q_shape"),
        ("_1", "k_proj", "kv_shape"),
        ("_2", "v_proj", "kv_shape"),
    ]:
        nodes.append(
            helper.make_node(
                "Reshape",
                [f"{ap}/{proj}/MatMul_output_0", f"{ap}/{shape_name}"],
                [f"{ap}/Reshape{tag}_output_0"],
                name=f"{ap}/Reshape{tag}",
            )
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{ap}/Reshape{tag}_output_0"],
                [f"{ap}/Transpose{tag}_output_0"],
                name=f"{ap}/Transpose{tag}",
                perm=[0, 2, 1, 3],
            )
        )

    qt = f"{ap}/Transpose_output_0"
    kt = f"{ap}/Transpose_1_output_0"
    vt = f"{ap}/Transpose_2_output_0"

    # --- KV cache concat ---
    nodes.append(
        helper.make_node("Concat", ["past_key_values.0.key", kt], ["present.0.key"], name=f"{ap}/Concat_5", axis=2)
    )
    nodes.append(
        helper.make_node("Concat", ["past_key_values.0.value", vt], ["present.0.value"], name=f"{ap}/Concat_6", axis=2)
    )

    # --- GQA repeat KV if needed ---
    if kv_heads != num_heads:
        reps = num_heads // kv_heads
        inits.extend(
            [
                _init(f"{ap}/rk/exp", np.array([1, reps, 1, 1], np.int64)),
                _init(f"{ap}/rk/ax", np.array([2], np.int64)),
                _init(f"{ap}/rk/rs", np.array([0, num_heads, -1, head_dim], np.int64)),
            ]
        )
        for t, src in [("k", "present.0.key"), ("v", "present.0.value")]:
            nodes.append(
                helper.make_node(
                    "Unsqueeze", [src, f"{ap}/rk/ax"], [f"{ap}/rk/{t}u"], name=f"{ap}/repeat_kv/{t}_unsqueeze"
                )
            )
            nodes.append(
                helper.make_node(
                    "Expand", [f"{ap}/rk/{t}u", f"{ap}/rk/exp"], [f"{ap}/rk/{t}e"], name=f"{ap}/repeat_kv/{t}_expand"
                )
            )
            nodes.append(
                helper.make_node(
                    "Reshape", [f"{ap}/rk/{t}e", f"{ap}/rk/rs"], [f"{ap}/rk/{t}r"], name=f"{ap}/repeat_kv/{t}_reshape"
                )
            )
        k_final, v_final = f"{ap}/rk/kr", f"{ap}/rk/vr"
    else:
        k_final, v_final = "present.0.key", "present.0.value"

    # --- Scaled dot-product attention ---
    nodes.append(
        helper.make_node(
            "Transpose", [k_final], [f"{ap}/Transpose_3_output_0"], name=f"{ap}/Transpose_3", perm=[0, 1, 3, 2]
        )
    )
    scale_val = float(np.float16(1.0 / head_dim**0.5))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            [f"{ap}/scale_output_0"],
            name=f"{ap}/scale",
            value=numpy_helper.from_array(np.array(scale_val, dtype=np.float16), name=""),
        )
    )
    nodes.append(helper.make_node("Mul", [qt, f"{ap}/scale_output_0"], [f"{ap}/Mul_8_output_0"], name=f"{ap}/Mul_8"))
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{ap}/Mul_8_output_0", f"{ap}/Transpose_3_output_0"],
            [f"{ap}/MatMul_output_0"],
            name=f"{ap}/MatMul",
        )
    )
    nodes.append(
        helper.make_node("Add", [f"{ap}/MatMul_output_0", "attn_bias"], [f"{ap}/Add_2_output_0"], name=f"{ap}/Add_2")
    )
    nodes.append(
        helper.make_node("Softmax", [f"{ap}/Add_2_output_0"], [f"{ap}/Softmax_output_0"], name=f"{ap}/Softmax", axis=-1)
    )
    nodes.append(
        helper.make_node(
            "MatMul", [f"{ap}/Softmax_output_0", v_final], [f"{ap}/MatMul_1_output_0"], name=f"{ap}/MatMul_1"
        )
    )

    # --- Transpose + Reshape back ---
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{ap}/MatMul_1_output_0"],
            [f"{ap}/Transpose_4_output_0"],
            name=f"{ap}/Transpose_4",
            perm=[0, 2, 1, 3],
        )
    )
    inits.append(_init(f"{ap}/out_rs", np.array([0, 0, hidden_size], np.int64)))
    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{ap}/Transpose_4_output_0", f"{ap}/out_rs"],
            [f"{ap}/Reshape_7_output_0"],
            name=f"{ap}/Reshape_7",
        )
    )

    # --- o_proj + residual + lm_head ---
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{ap}/Reshape_7_output_0", "model.layers.0.self_attn.o_proj.weight"],
            [f"{ap}/o_proj/MatMul_output_0"],
            name=f"{ap}/o_proj/MatMul",
        )
    )
    hidden_out = "/model/embed_tokens/Gather_output_0"
    nodes.append(
        helper.make_node(
            "Add",
            [hidden_out, f"{ap}/o_proj/MatMul_output_0"],
            ["/model/layers.0/residual_output_0"],
            name="/model/layers.0/residual_add",
        )
    )
    inits.append(_init("lm_head.weight", _fp16(hidden_size, VOCAB_SIZE)))
    nodes.append(
        helper.make_node(
            "MatMul", ["/model/layers.0/residual_output_0", "lm_head.weight"], ["logits"], name="/lm_head/MatMul"
        )
    )

    graph = helper.make_graph(nodes, "llm_attn", graph_inputs, graph_outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _build_quantized_model() -> onnx.ModelProto:
    """Create a quantized model with DequantizeLinear(8x16) feeding into MatMul."""
    qweight = numpy_helper.from_array(np.random.randint(-128, 127, (8, 16), dtype=np.int8), "qweight")
    scale = numpy_helper.from_array(np.array([0.01], dtype=np.float32), "scale")
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
    y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])
    nodes = [
        helper.make_node("DequantizeLinear", ["qweight", "scale"], ["dq_out"], name="dql_0"),
        helper.make_node("MatMul", ["X", "dq_out"], ["Y"], name="matmul_0"),
    ]
    graph = helper.make_graph(nodes, "quant_model", [x_input], [y_output], initializer=[qweight, scale])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# Tests -- call the REAL surgery through Olive pass
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Disabled pending onnx-graphsurgeon compatibility with onnx>=1.20")
def test_replace_gqa(tmp_path):
    """Run real replace-gqa surgery through Olive pass, verify output structure."""
    hidden, heads, kv, hd = _get_config()
    before_proto = _build_attention_model(hidden, heads, kv, hd)

    model_path = str(tmp_path / "model.onnx")
    onnx.save(before_proto, model_path)
    ov_model = ONNXModelHandler(model_path=model_path)

    config = {
        "surgery_type": "replace-gqa",
        "surgery_params": {
            "hf_model_id": MODEL_ID,
            "max_seq_len": 128,
            "io_dtype": "float16",
        },
    }

    p = create_pass_from_dict(NVModelOptGraphSurgery, config, disable_search=True)
    result = p.run(ov_model, str(tmp_path / "output_gqa"))
    result_proto = result.load_model()

    op_types = [n.op_type for n in result_proto.graph.node]
    node_names = [n.name for n in result_proto.graph.node]
    input_names = [i.name for i in result_proto.graph.input]
    output_names = [o.name for o in result_proto.graph.output]

    assert "GroupQueryAttention" in op_types, f"Expected GQA node, got ops: {op_types}"
    assert any("o_proj" in n for n in node_names), "o_proj MatMul should be preserved"
    assert any("past_key_values" in n for n in input_names), f"Expected KV cache input, got: {input_names}"
    assert any("present" in n for n in output_names), f"Expected present output, got: {output_names}"
    assert "Softmax" not in op_types, "Old Softmax should be removed"
    assert any("qkv_proj" in n for n in node_names), "Expected fused QKV MatMul"
    assert "ReduceSum" in op_types, "Attention mask subgraph should be present"

    gqa_node = next(n for n in result_proto.graph.node if n.op_type == "GroupQueryAttention")
    attrs = {a.name: (a.i if a.type == 2 else a.f) for a in gqa_node.attribute}
    assert attrs["num_heads"] == heads
    assert attrs["kv_num_heads"] == kv
    assert attrs["do_rotary"] == 1


@pytest.mark.skip(reason="Disabled pending onnx-graphsurgeon compatibility with onnx>=1.20")
def test_transpose_dq(tmp_path):
    """Run real transpose-dq surgery through Olive pass, verify transposed weights."""
    before_proto = _build_quantized_model()
    model_path = str(tmp_path / "model_quant.onnx")
    onnx.save(before_proto, model_path)
    ov_model = ONNXModelHandler(model_path=model_path)

    config = {
        "surgery_type": "transpose-dq",
        "surgery_params": {},
    }

    p = create_pass_from_dict(NVModelOptGraphSurgery, config, disable_search=True)
    result = p.run(ov_model, str(tmp_path / "output_dq"))
    result_proto = result.load_model()

    op_types = [n.op_type for n in result_proto.graph.node]
    node_names = [n.name for n in result_proto.graph.node]

    assert "DequantizeLinear" in op_types, f"DequantizeLinear should still exist, got: {op_types}"
    assert "Transpose" in op_types, f"Expected Transpose node, got: {op_types}"
    assert any("transpose_back" in n for n in node_names), f"Expected *_transpose_back, got: {node_names}"
    assert "MatMul" in op_types, "MatMul should still exist"

    for init in result_proto.graph.initializer:
        if "transposed" in init.name:
            assert list(init.dims) == [16, 8], f"Expected transposed shape [16,8], got {list(init.dims)}"
