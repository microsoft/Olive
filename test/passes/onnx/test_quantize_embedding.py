# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
from onnx import TensorProto, helper, numpy_helper

from olive.passes.onnx.graph_surgeries import QuantizeEmbeddingInt8, ShareEmbeddingLmHead


def _make_model_with_fp16_embed(vocab_size=64, hidden_size=64, block_size=32):
    """Create a minimal ONNX model with FP16 Gather embedding and INT4 MatMulNBits lm_head."""
    # Embedding: Gather with FP16 weight
    embed_weight = np.random.randn(vocab_size, hidden_size).astype(np.float16)
    embed_init = numpy_helper.from_array(embed_weight, name="model.embed_tokens.weight")

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "seq_len"])

    gather_node = helper.make_node(
        "Gather",
        inputs=["model.embed_tokens.weight", "input_ids"],
        outputs=["embed_output"],
        name="/model/embed_tokens/Gather",
    )

    # lm_head: MatMulNBits with INT4 weight
    num_blocks = hidden_size // block_size
    lm_weight = np.random.randint(0, 255, (vocab_size, num_blocks, block_size // 2), dtype=np.uint8)
    lm_scales = np.random.randn(vocab_size, num_blocks).astype(np.float16) * 0.01
    lm_zp = np.full((vocab_size, num_blocks), 8, dtype=np.uint8)

    lm_weight_init = numpy_helper.from_array(lm_weight, name="lm_head.MatMul_Q4.qweight")
    lm_scales_init = numpy_helper.from_array(lm_scales, name="lm_head.MatMul_Q4.scales")
    lm_zp_init = numpy_helper.from_array(lm_zp, name="lm_head.MatMul_Q4.zp")

    lm_head_node = helper.make_node(
        "MatMulNBits",
        inputs=["embed_output", "lm_head.MatMul_Q4.qweight", "lm_head.MatMul_Q4.scales", "lm_head.MatMul_Q4.zp"],
        outputs=["logits"],
        name="/lm_head/MatMulNBits",
        domain="com.microsoft",
        bits=4,
        block_size=block_size,
        K=hidden_size,
        N=vocab_size,
    )

    graph = helper.make_graph(
        [gather_node, lm_head_node],
        "test",
        [input_ids],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT16, ["batch_size", "seq_len", vocab_size])],
        initializer=[embed_init, lm_weight_init, lm_scales_init, lm_zp_init],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )


class TestQuantizeEmbeddingInt8:
    def test_replaces_gather_with_gbq(self):
        model = _make_model_with_fp16_embed()
        surgery = QuantizeEmbeddingInt8()
        result = surgery(model)

        # Verify Gather is replaced with GatherBlockQuantized
        node_types = [n.op_type for n in result.graph.node]
        assert "Gather" not in node_types or all(
            "embed_tokens" not in n.name for n in result.graph.node if n.op_type == "Gather"
        )
        gbq_nodes = [n for n in result.graph.node if n.op_type == "GatherBlockQuantized"]
        assert len(gbq_nodes) == 1

        gbq = gbq_nodes[0]
        attrs = {a.name: a.i for a in gbq.attribute}
        assert attrs["bits"] == 8
        assert attrs["block_size"] == 32

        # Verify zero_point input exists (4 inputs: weight, input_ids, scales, zp)
        assert len(gbq.input) == 4

    def test_reduces_weight_size(self):
        model = _make_model_with_fp16_embed(vocab_size=256, hidden_size=128)
        surgery = QuantizeEmbeddingInt8()

        result = surgery(model)

        # FP16 weight should be removed
        fp16_names = [init.name for init in result.graph.initializer if init.name == "model.embed_tokens.weight"]
        assert len(fp16_names) == 0

        # INT8 weight should exist
        int8_names = [init.name for init in result.graph.initializer if "_Q8" in init.name]
        assert len(int8_names) == 1

    def test_skips_non_fp16(self):
        model = _make_model_with_fp16_embed()
        surgery = QuantizeEmbeddingInt8()

        # First pass: quantize to INT8
        result1 = surgery(model)
        # Second pass: should skip (already quantized)
        result2 = surgery(result1)

        # Should still have exactly 1 GBQ node
        gbq_count = sum(1 for n in result2.graph.node if n.op_type == "GatherBlockQuantized")
        assert gbq_count == 1

    def test_skips_when_hidden_not_divisible(self):
        # hidden_size=33, not divisible by block_size=32
        embed_weight = np.random.randn(64, 33).astype(np.float16)
        embed_init = numpy_helper.from_array(embed_weight, name="model.embed_tokens.weight")
        input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1])
        output = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 33])
        gather = helper.make_node(
            "Gather", ["model.embed_tokens.weight", "input_ids"], ["out"], name="/model/embed_tokens/Gather"
        )
        graph = helper.make_graph([gather], "test", [input_ids], [output], initializer=[embed_init])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
        )

        surgery = QuantizeEmbeddingInt8()
        result = surgery(model)

        # Should still have Gather (not replaced)
        assert any(n.op_type == "Gather" for n in result.graph.node)


class TestShareEmbeddingLmHead:
    def test_shares_weight(self):
        model = _make_model_with_fp16_embed()

        # First quantize embedding to INT8
        q_surgery = QuantizeEmbeddingInt8()
        model = q_surgery(model)

        # Then share
        s_surgery = ShareEmbeddingLmHead()
        result = s_surgery(model)

        # lm_head should now be INT8
        lm_head = next(n for n in result.graph.node if n.op_type == "MatMulNBits" and "lm_head" in n.name)
        attrs = {a.name: a.i for a in lm_head.attribute}
        assert attrs["bits"] == 8

        # Should have a Reshape node for weight sharing
        reshape_nodes = [n for n in result.graph.node if "Reshape_shared" in n.name]
        assert len(reshape_nodes) == 1

        # Reshape should reference the embedding weight
        reshape = reshape_nodes[0]
        assert "embed_tokens" in reshape.input[0]

        # lm_head should use shared scales
        assert "embed_tokens" in lm_head.input[2]  # scales

    def test_idempotent(self):
        model = _make_model_with_fp16_embed()
        q_surgery = QuantizeEmbeddingInt8()
        model = q_surgery(model)

        s_surgery = ShareEmbeddingLmHead()
        result1 = s_surgery(model)
        # Applying again should be a no-op
        result2 = s_surgery(result1)

        # Should still have exactly 1 Reshape_shared node
        reshape_count = sum(1 for n in result2.graph.node if "Reshape_shared" in n.name)
        assert reshape_count == 1

    def test_skips_without_gbq(self):
        model = _make_model_with_fp16_embed()
        # Don't quantize embedding first
        s_surgery = ShareEmbeddingLmHead()
        result = s_surgery(model)

        # Should be unchanged — still has Gather
        assert any(n.op_type == "Gather" for n in result.graph.node)

    def test_removes_old_lm_head_weights(self):
        model = _make_model_with_fp16_embed()
        q_surgery = QuantizeEmbeddingInt8()
        model = q_surgery(model)

        s_surgery = ShareEmbeddingLmHead()
        result = s_surgery(model)

        new_init_names = {init.name for init in result.graph.initializer}

        # Old lm_head weights should be removed
        assert "lm_head.MatMul_Q4.qweight" not in new_init_names
        assert "lm_head.MatMul_Q4.scales" not in new_init_names
        assert "lm_head.MatMul_Q4.zp" not in new_init_names
