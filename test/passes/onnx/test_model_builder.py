# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import onnx
import pytest

from olive.model import CompositeModelHandler, HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_builder import ModelBuilder
from olive.passes.pytorch.rtn import Rtn
from test.utils import make_local_tiny_llama

TINY_RANDOM_LLAMA_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _create_test_onnx_model(model_path: Path, node_name: str):
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1])
    node = onnx.helper.make_node("Identity", ["input"], ["output"], name=node_name)
    graph = onnx.helper.make_graph([node], "test_graph", [input_info], [output_info])
    model = onnx.helper.make_model(graph)
    onnx.save(model, model_path)


def _mock_genai_builder(monkeypatch, create_model_fn, check_extra_options_fn=None):
    builder_module = types.ModuleType("onnxruntime_genai.models.builder")
    builder_module.create_model = create_model_fn
    builder_module.check_extra_options = check_extra_options_fn or (lambda *args, **kwargs: None)
    models_module = types.ModuleType("onnxruntime_genai.models")
    models_module.builder = builder_module
    genai_module = types.ModuleType("onnxruntime_genai")
    genai_module.__version__ = "0.15.0"
    genai_module.models = models_module
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", genai_module)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.models", models_module)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.models.builder", builder_module)
    monkeypatch.setattr(ModelBuilder, "maybe_patch_quant", staticmethod(lambda: None))


@pytest.mark.parametrize("metadata_only", [True, False])
def test_model_builder(tmp_path, metadata_only):
    input_model = make_local_tiny_llama(tmp_path / "input_model", "onnx" if metadata_only else "hf")

    p = create_pass_from_dict(
        ModelBuilder,
        {"precision": "fp32", "metadata_only": metadata_only, "extra_options": {"int4_is_symmetric": True}},
        disable_search=True,
    )
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    assert Path(output_folder / "genai_config.json").exists()


@pytest.mark.parametrize("embeds", [True, False])
@pytest.mark.parametrize("group_size", [16, -1])
def test_model_builder_olive_quant(tmp_path, embeds, group_size):
    # set up quantized model
    input_model = create_pass_from_dict(
        Rtn,
        {
            "bits": 4,
            "group_size": group_size,
            "symmetric": False,
            "lm_head": True,
            "embeds": embeds,
        },
        disable_search=True,
    ).run(
        make_local_tiny_llama(tmp_path / "hf_model", "hf"),
        tmp_path / "quantized_model",
    )

    p = create_pass_from_dict(ModelBuilder, {"precision": "int4"}, disable_search=True)
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    assert Path(output_folder / "genai_config.json").exists()


@pytest.mark.parametrize("layer_annotations", [True, False])
def test_model_builder_layer_annotations(tmp_path, layer_annotations):
    """Test that layer annotations are correctly applied to the output ONNX model."""
    input_model = make_local_tiny_llama(tmp_path / "input_model", "hf")

    if layer_annotations:
        # Create layer annotations to be applied
        # Keys are layer names, values are lists of node-name substrings to match
        annotations = {
            "embedding_layer": ["embed_tokens"],
            "norm_layer": ["norm"],
        }
        input_model.model_attributes = {"layer_annotations": annotations}

    p = create_pass_from_dict(
        ModelBuilder,
        {"precision": "fp32"},
        disable_search=True,
    )
    output_folder = tmp_path / "output_model"

    # execute the pass
    output_model = p.run(input_model, output_folder)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()

    if layer_annotations:
        # Verify that metadata properties were applied to nodes
        model_proto = onnx.load(output_model.model_path, load_external_data=False)
        node_names_with_metadata = {node.name for node in model_proto.graph.node if node.metadata_props}
        assert len(node_names_with_metadata) > 0, (
            "Expected nodes with metadata_props when layer_annotations are provided"
        )


def test_model_builder_uses_saved_test_model_path(tmp_path):
    test_model_path = tmp_path / "saved_test_model"
    output_folder = tmp_path / "output_model"

    mock_cfg = MagicMock()
    mock_cfg.to_dict.return_value = {}
    with patch.object(HfModelHandler, "get_hf_model_config", return_value=mock_cfg):
        input_model = HfModelHandler(
            model_path=TINY_RANDOM_LLAMA_MODEL_ID,
            test_model_config={"hidden_layers": 2},
            test_model_path=str(test_model_path),
        )

    def materialize_test_model(*args, **kwargs):
        test_model_path.mkdir(parents=True, exist_ok=True)
        (test_model_path / "config.json").write_text("{}")
        return MagicMock()

    def fake_create_model(*_, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / kwargs["filename"]).write_text("dummy onnx file")
        (output_dir / "genai_config.json").write_text("{}")

    fake_builder = types.ModuleType("onnxruntime_genai.models.builder")
    fake_builder.create_model = MagicMock(side_effect=fake_create_model)
    fake_builder.check_extra_options = MagicMock()
    fake_models = types.ModuleType("onnxruntime_genai.models")
    fake_models.builder = fake_builder
    fake_ort_genai = types.ModuleType("onnxruntime_genai")
    fake_ort_genai.models = fake_models
    fake_ort_genai.__version__ = "0.0.0"

    p = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True)

    with (
        patch.object(ModelBuilder, "maybe_patch_quant"),
        patch.dict(
            sys.modules,
            {
                "onnxruntime_genai": fake_ort_genai,
                "onnxruntime_genai.models": fake_models,
                "onnxruntime_genai.models.builder": fake_builder,
            },
        ),
        patch.object(input_model, "load_model", side_effect=materialize_test_model) as mock_load_model,
        patch.object(input_model, "save_metadata", return_value=[]),
    ):
        output_model = p.run(input_model, output_folder)

    assert isinstance(output_model, ONNXModelHandler)
    assert mock_load_model.call_count == 1
    assert Path(output_model.model_path).exists()
    assert test_model_path.exists()
    assert fake_builder.create_model.call_args.kwargs["model_name"] == str(test_model_path)
    assert fake_builder.create_model.call_args.kwargs["input_path"] == str(test_model_path)


def test_model_builder_materializes_weights_for_config_only_test_dir(tmp_path):
    """A config-only test-model dir (no weights) must still trigger weight materialization.

    Otherwise the model builder would initialize its own weights and the ONNX model would not
    match the reference model that OnnxDiscrepancyCheck later loads from the same directory.
    """
    from olive.common.hf.utils import TEST_MODEL_MARKER_FILE

    test_model_path = tmp_path / "reference_hf_model"
    output_folder = tmp_path / "output_model"

    # Pre-create a config-only Olive test-model directory: marker + config.json, but no weights.
    test_model_path.mkdir(parents=True, exist_ok=True)
    (test_model_path / "config.json").write_text("{}")
    (test_model_path / TEST_MODEL_MARKER_FILE).write_text(
        json.dumps({"type": "olive_hf_test_model", "test_model_config": {}})
    )

    mock_cfg = MagicMock()
    mock_cfg.to_dict.return_value = {}
    with patch.object(HfModelHandler, "get_hf_model_config", return_value=mock_cfg):
        input_model = HfModelHandler(
            model_path=TINY_RANDOM_LLAMA_MODEL_ID,
            test_model_config={"hidden_layers": 2},
            test_model_path=str(test_model_path),
        )

    def materialize_weights(*args, **kwargs):
        (test_model_path / "model.safetensors").write_text("weights")
        return MagicMock()

    def fake_create_model(*_, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / kwargs["filename"]).write_text("dummy onnx file")
        (output_dir / "genai_config.json").write_text("{}")

    fake_builder = types.ModuleType("onnxruntime_genai.models.builder")
    fake_builder.create_model = MagicMock(side_effect=fake_create_model)
    fake_builder.check_extra_options = MagicMock()
    fake_models = types.ModuleType("onnxruntime_genai.models")
    fake_models.builder = fake_builder
    fake_ort_genai = types.ModuleType("onnxruntime_genai")
    fake_ort_genai.models = fake_models
    fake_ort_genai.__version__ = "0.0.0"

    p = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True)

    with (
        patch.object(ModelBuilder, "maybe_patch_quant"),
        patch.dict(
            sys.modules,
            {
                "onnxruntime_genai": fake_ort_genai,
                "onnxruntime_genai.models": fake_models,
                "onnxruntime_genai.models.builder": fake_builder,
            },
        ),
        patch.object(input_model, "load_model", side_effect=materialize_weights) as mock_load_model,
        patch.object(input_model, "save_metadata", return_value=[]),
    ):
        output_model = p.run(input_model, output_folder)

    assert isinstance(output_model, ONNXModelHandler)
    # Weights were missing, so load_model must be called to persist them into the shared dir.
    assert mock_load_model.call_count == 1
    # The ONNX model is built from the shared test-model directory (same weights as the reference).
    assert fake_builder.create_model.call_args.kwargs["input_path"] == str(test_model_path.resolve())


def test_model_builder_apply_annotations_on_single_file_fallback(tmp_path, monkeypatch):
    def fake_create_model(
        model_name, input_path, output_dir, precision, execution_provider, cache_dir, filename, **kwargs
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _create_test_onnx_model(output_dir / "actual.onnx", "test_node")
        (output_dir / "actual.onnx.data").write_text("external_data")
        (output_dir / "tokenizer.json").write_text("{}")
        (output_dir / "genai_config.json").write_text(json.dumps({"search": {}}))

    _mock_genai_builder(monkeypatch, fake_create_model)
    input_model = Mock(spec=HfModelHandler)
    input_model.model_name_or_path = "dummy-model"
    input_model.adapter_path = None
    input_model.test_model_config = None
    input_model.test_model_path = None
    input_model.model_attributes = {"split_assignments": {"model.layers.0": 1}}

    p = create_pass_from_dict(
        ModelBuilder, {"precision": "fp32", "extra_options": {"filename": "expected.onnx"}}, disable_search=True
    )
    output_folder = tmp_path / "output_model"
    output_model = p.run(input_model, output_folder)

    assert isinstance(output_model, ONNXModelHandler)
    assert output_model.onnx_file_name == "actual.onnx"
    model_proto = onnx.load(output_folder / "actual.onnx", load_external_data=False)
    metadata_props = {prop.key: prop.value for prop in model_proto.metadata_props}
    assert metadata_props["split_assignments"] == "model.layers.0=1"
    assert str(output_folder / "actual.onnx") not in output_model.model_attributes["additional_files"]
    assert str(output_folder / "actual.onnx.data") not in output_model.model_attributes["additional_files"]
    assert str(output_folder / "tokenizer.json") in output_model.model_attributes["additional_files"]


def test_model_builder_multi_file_output_preserves_component_filenames(tmp_path, monkeypatch):
    def fake_create_model(
        model_name, input_path, output_dir, precision, execution_provider, cache_dir, filename, **kwargs
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _create_test_onnx_model(output_dir / "encoder.onnx", "encoder_node")
        _create_test_onnx_model(output_dir / "decoder.onnx", "decoder_node")
        (output_dir / "encoder.onnx.data").write_text("encoder_data")
        (output_dir / "decoder.onnx.data").write_text("decoder_data")
        (output_dir / "tokenizer.json").write_text("{}")
        (output_dir / "genai_config.json").write_text(json.dumps({"search": {}}))

    _mock_genai_builder(monkeypatch, fake_create_model)
    input_model = Mock(spec=HfModelHandler)
    input_model.model_name_or_path = "dummy-model"
    input_model.adapter_path = None
    input_model.test_model_config = None
    input_model.test_model_path = None
    input_model.model_attributes = {}

    p = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True)
    output_folder = tmp_path / "output_model"
    output_model = p.run(input_model, output_folder)

    assert isinstance(output_model, CompositeModelHandler)
    expected_component_names = sorted(["encoder.onnx", "decoder.onnx"])
    assert output_model.model_component_names == expected_component_names
    component_onnx_files = [component.onnx_file_name for component in output_model.model_components]
    assert component_onnx_files == output_model.model_component_names
    additional_files = output_model.model_attributes["additional_files"]
    assert str(output_folder / "encoder.onnx") not in additional_files
    assert str(output_folder / "decoder.onnx") not in additional_files
    assert str(output_folder / "encoder.onnx.data") not in additional_files
    assert str(output_folder / "decoder.onnx.data") not in additional_files
    assert str(output_folder / "tokenizer.json") in additional_files


def test_olive_quantized_model_raises_for_moe():
    """ModelBuilder must reject Olive-quantized MoE checkpoints.

    Errors out cleanly so the user reaches for an alternative builder
    or re-runs RTN without ``moe=True``.
    """
    from olive.passes.onnx.model_builder import OliveQuantizedModel

    quant_attrs = {
        "config": {
            "bits": 4,
            "group_size": 32,
            "symmetric": True,
            "embeds": False,
            "lm_head": False,
            "tie_word_embeddings": False,
            "moe": True,
            "overrides": {},
        }
    }
    with pytest.raises(NotImplementedError, match="MoE"):
        OliveQuantizedModel(
            quant_type="olive",
            input_path="/tmp/does_not_matter",
            quant_attrs=quant_attrs,
            q_size=64,
            kv_size=64,
            intermediate_size=64,
            num_layers=1,
        )


def test_olive_quantized_model_migrates_non_moe_keys(tmp_path):
    """M7 regression: ``set_tensor``'s non-MoE key migration must be correct.

    It must correctly map Olive's ``<pname>_qweight`` / ``_scales`` / ``_qzeros`` naming
    onto ``QuantizedTensorModule``'s bare ``qweight`` / ``scales`` / ``qzeros`` attributes,
    with correct ``in_features`` / ``out_features`` / block reshape -- previously only the
    ``moe=True``-rejection path had coverage for this code.
    """
    from olive.passes.onnx.model_builder import OliveQuantizedModel

    # Produce a real Olive-quantized (non-MoE) checkpoint via the actual Rtn pass.
    input_model = make_local_tiny_llama(tmp_path / "hf_model", "hf")
    quantized_model = create_pass_from_dict(
        Rtn,
        {
            "bits": 4,
            "group_size": 16,
            "symmetric": False,
            "lm_head": True,
            "embeds": True,
        },
        disable_search=True,
    ).run(input_model, tmp_path / "quantized_model")

    loaded = quantized_model.load_model()
    qcfg = loaded.config.quantization_config.to_dict()

    quant_attrs = {
        "config": {
            "bits": qcfg["bits"],
            "group_size": qcfg["group_size"],
            "symmetric": qcfg["symmetric"],
            "embeds": qcfg["embeds"],
            "lm_head": qcfg["lm_head"],
            "tie_word_embeddings": qcfg["tie_word_embeddings"],
            "moe": qcfg["moe"],
            "overrides": qcfg.get("overrides") or {},
        }
    }
    hidden_size = loaded.config.hidden_size
    num_heads = loaded.config.num_attention_heads
    num_kv_heads = getattr(loaded.config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    model = OliveQuantizedModel(
        quant_type="olive",
        input_path=quantized_model.model_path,
        quant_attrs=quant_attrs,
        q_size=hidden_size,
        kv_size=num_kv_heads * head_dim,
        intermediate_size=loaded.config.intermediate_size,
        num_layers=loaded.config.num_hidden_layers,
    )

    q_proj = model.layers[0].self_attn.q_proj
    assert q_proj.qweight is not None
    assert q_proj.scales is not None
    assert q_proj.bits == 4
    assert q_proj.in_features == hidden_size
    assert q_proj.out_features == hidden_size
    # qweight reshaped to (out_features, num_blocks, blob_size)
    assert q_proj.qweight.dim() == 3
    assert q_proj.qweight.shape[0] == hidden_size

    down_proj = model.layers[0].mlp.down_proj
    assert down_proj.qweight is not None
    assert down_proj.bits == 4
    assert down_proj.in_features == loaded.config.intermediate_size
    assert down_proj.out_features == hidden_size


def test_olive_quantized_model_applies_regex_overrides(tmp_path):
    """``re:``-prefixed override keys must be honored by ModelBuilder.

    ``overrides`` keys are documented (``olive.common.quant.patterns``) to support ``re:``
    regex patterns matched with ``re.fullmatch``. ModelBuilder used to look them up with a
    plain ``dict.get``, so a regex-keyed override silently fell back to the global
    ``bits``/``group_size`` -- which then miscomputes ``in_features`` and reshapes the packed
    ``qweight`` incorrectly.
    """
    from olive.passes.onnx.model_builder import OliveQuantizedModel

    default_bits, default_group_size = 4, 16
    override_bits, override_group_size = 8, 32
    override_key = r"re:model\.layers\.0\.mlp\.down_proj"

    input_model = make_local_tiny_llama(tmp_path / "hf_model", "hf")
    quantized_model = create_pass_from_dict(
        Rtn,
        {
            "bits": default_bits,
            "group_size": default_group_size,
            "symmetric": False,
            "overrides": {override_key: {"bits": override_bits, "group_size": override_group_size}},
        },
        disable_search=True,
    ).run(input_model, tmp_path / "quantized_model")

    loaded = quantized_model.load_model()
    qcfg = loaded.config.quantization_config.to_dict()
    # The regex key must survive serialization, otherwise this test would pass vacuously.
    assert override_key in (qcfg.get("overrides") or {})
    assert loaded.config.num_hidden_layers > 1, "need a second layer to check the non-matched case"

    hidden_size = loaded.config.hidden_size
    num_heads = loaded.config.num_attention_heads
    num_kv_heads = getattr(loaded.config, "num_key_value_heads", num_heads)
    model = OliveQuantizedModel(
        quant_type="olive",
        input_path=quantized_model.model_path,
        quant_attrs={
            "config": {
                "bits": qcfg["bits"],
                "group_size": qcfg["group_size"],
                "symmetric": qcfg["symmetric"],
                "embeds": qcfg["embeds"],
                "lm_head": qcfg["lm_head"],
                "tie_word_embeddings": qcfg["tie_word_embeddings"],
                "moe": qcfg["moe"],
                "overrides": qcfg.get("overrides") or {},
            }
        },
        q_size=hidden_size,
        kv_size=num_kv_heads * (hidden_size // num_heads),
        intermediate_size=loaded.config.intermediate_size,
        num_layers=loaded.config.num_hidden_layers,
    )

    # Matched layer -> overridden bits / group_size (and therefore correct in_features).
    matched = model.layers[0].mlp.down_proj
    assert matched.bits == override_bits
    assert matched.group_size == override_group_size
    assert matched.in_features == loaded.config.intermediate_size
    assert matched.qweight.shape == (
        hidden_size,
        loaded.config.intermediate_size // override_group_size,
        override_group_size * override_bits // 8,
    )

    # Non-matched layer -> pass-level defaults.
    unmatched = model.layers[1].mlp.down_proj
    assert unmatched.bits == default_bits
    assert unmatched.group_size == default_group_size
    assert unmatched.in_features == loaded.config.intermediate_size
    assert unmatched.qweight.shape == (
        hidden_size,
        loaded.config.intermediate_size // default_group_size,
        default_group_size * default_bits // 8,
    )


def test_model_builder_prechecks_extra_options(tmp_path, monkeypatch):
    def fake_check_extra_options(
        model_name, input_path, output_dir, precision, execution_provider, cache_dir, extra_options
    ):
        assert model_name == "dummy-model"
        assert input_path == "dummy-model"
        assert output_dir == str(tmp_path / "output_model")
        assert precision == "fp32"
        assert execution_provider == "cpu"
        assert cache_dir
        # Values are serialized the way `--extra_options key=value` would produce them.
        assert extra_options["exclude_embeds"] == "true"
        assert extra_options["use_qdq"] == "false"
        assert extra_options["int4_op_types_to_quantize"] == "MatMul/Gather"
        assert extra_options["int4_nodes_to_exclude"] == "node_1,node_2"
        # An option the model builder does not treat as a list is left alone.
        assert extra_options["int4_block_size"] == 32
        extra_options["hf_details"] = {
            "extra_kwargs": {},
            "hf_name": model_name,
            "hf_config": Mock(),
        }

    def fake_create_model(
        model_name, input_path, output_dir, precision, execution_provider, cache_dir, filename, **kwargs
    ):
        assert "hf_details" in kwargs
        output_dir = Path(output_dir)
        _create_test_onnx_model(output_dir / filename, "test_node")
        (output_dir / "genai_config.json").write_text(json.dumps({"search": {}}))

    _mock_genai_builder(monkeypatch, fake_create_model, fake_check_extra_options)

    input_model = Mock(spec=HfModelHandler)
    input_model.model_name_or_path = "dummy-model"
    input_model.adapter_path = None
    input_model.test_model_config = None
    input_model.test_model_path = None
    input_model.model_attributes = {}

    p = create_pass_from_dict(
        ModelBuilder,
        {
            "precision": "fp32",
            "exclude_embeds": True,
            "use_qdq": False,
            "int4_block_size": 32,
            "int4_op_types_to_quantize": ["MatMul", "Gather"],
            "int4_nodes_to_exclude": ["node_1", "node_2"],
        },
        disable_search=True,
    )
    output_model = p.run(input_model, tmp_path / "output_model")

    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
