# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Network-free Olive -> Mobius split-export regression for tied Gemma4 weights."""

import json

import numpy as np
import pytest
import torch

from olive.model import HfModelHandler
from olive.workflows import run as olive_run
from test.passes.pytorch.test_quantization_utils import make_local_tiny_tied_gemma4


def test_tied_gemma4_checkpoint_exports_and_runs_both_onnx_components(tmp_path):
    pytest.importorskip("transformers.models.gemma4")
    mobius = pytest.importorskip("mobius")
    if not hasattr(mobius, "SharedWeightInfo"):
        pytest.skip("Requires Mobius cross-component shared-weight support")
    ir = pytest.importorskip("onnx_ir")
    ort = pytest.importorskip("onnxruntime", minversion="1.28")

    source = tmp_path / "source"
    make_local_tiny_tied_gemma4(source)
    assembled = tmp_path / "assembled"
    olive_run(
        {
            "input_model": {
                "type": "HfModel",
                "config": {"model_path": str(source), "task": "image-text-to-text"},
            },
            "passes": {
                "decoder_kquant": {
                    "type": "KQuant",
                    "bits": 4,
                    "group_size": 16,
                    "sym": True,
                    "overrides": {"lm_head": {"bits": 8}},
                },
                "embedding_kquant": {"type": "KQuant", "bits": 8, "group_size": 16, "sym": True},
            },
            "builds": {
                "decoder": {"components": ["decoder"], "pipeline": ["decoder_kquant"]},
                "embedding": {"components": ["embedding"], "pipeline": ["embedding_kquant"]},
            },
            "engine": {
                "output_dir": str(assembled),
                "cache_dir": str(tmp_path / "cache"),
                "evaluate_input_model": False,
            },
            "max_concurrent_builds": 1,
        }
    )

    config = json.loads((assembled / "config.json").read_text(encoding="utf-8"))
    index = json.loads((assembled / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["tie_word_embeddings"] is True
    assert "model.language_model.embed_tokens.weight_qweight" in index["weight_map"]
    assert "lm_head.weight_qweight" not in index["weight_map"]

    package = mobius.build(str(assembled), dtype="f32")
    sessions = {}
    for component in ("embedding", "decoder"):
        path = tmp_path / f"{component}.onnx"
        ir.save(package[component], path, external_data=f"{component}.onnx.data")
        sessions[component] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    ids = np.asarray([[2, 10]], dtype=np.int64)
    embedding = sessions["embedding"]
    embedding_outputs = dict(
        zip(
            (output.name for output in embedding.get_outputs()),
            embedding.run(None, {"input_ids": ids, "image_features": np.zeros((0, 64), dtype=np.float32)}),
        )
    )
    decoder = sessions["decoder"]
    feeds = {
        "inputs_embeds": embedding_outputs["inputs_embeds"],
        "per_layer_inputs": embedding_outputs["per_layer_inputs"],
        "attention_mask": np.ones((1, 2), dtype=np.int64),
        "position_ids": np.asarray([[0, 1]], dtype=np.int64),
    }
    for input_info in decoder.get_inputs():
        if input_info.name.startswith("past_key_values."):
            feeds[input_info.name] = np.zeros((1, 2, 0, 16), dtype=np.float32)
    logits = decoder.run(["logits"], feeds)[0]

    hf_model = HfModelHandler(model_path=str(assembled), task="image-text-to-text").load_model().eval()
    with torch.no_grad():
        reference = (
            hf_model(
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.ones_like(torch.from_numpy(ids)),
                use_cache=False,
            )
            .logits.float()
            .numpy()
        )
    np.testing.assert_allclose(logits, reference, rtol=1e-3, atol=1e-3)
