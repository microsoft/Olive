# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto, helper

from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import update_llm_pipeline_genai_config
from olive.passes.onnx.compose import ComposeOnnxModels
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.model_builder import ModelBuilder
from olive.passes.onnx.split import SplitModel
from olive.passes.onnx.static_llm import StaticLLM
from test.utils import make_local_tiny_llama


@pytest.mark.parametrize(
    "use_mb",
    [
        True,
        pytest.param(False, marks=pytest.mark.skip(reason="Dynamo export fails for Llama, need fix")),
    ],
)
def test_compose_onnx_models_composite(tmp_path, use_mb):
    # setup
    pytorch_model = make_local_tiny_llama(tmp_path)
    onnx_model = create_pass_from_dict(
        ModelBuilder if use_mb else OnnxConversion,
        {"precision": "fp32"} if use_mb else {"use_dynamo_exporter": True},
        disable_search=True,
    ).run(pytorch_model, tmp_path / "onnx_model")
    split_model = create_pass_from_dict(
        SplitModel,
        {
            "split_assignments": {
                "model.embed_tokens": 0,
                "model.layers.0": 1,
                "model.layers.1": 2,
                "lm_head": 3,
            }
        },
        disable_search=True,
    ).run(onnx_model, tmp_path / "split_model")

    p = create_pass_from_dict(ComposeOnnxModels, {}, disable_search=True)

    # execute
    output_model_path = tmp_path / "output_model"
    output_model = p.run(split_model, output_model_path)

    # check
    assert isinstance(output_model, ONNXModelHandler)
    assert Path(output_model.model_path).exists()
    expected_io_config = onnx_model.io_config
    actual_io_config = output_model.io_config
    for io_key in ["input", "output"]:
        # check the i/o names match
        assert set(actual_io_config[f"{io_key}_names"]) == set(expected_io_config[f"{io_key}_names"])
        # check the i/o shapes/types match
        for info_key in ["shapes", "types"]:
            expected_info_dict = dict(
                zip(expected_io_config[f"{io_key}_names"], expected_io_config[f"{io_key}_{info_key}"])
            )
            actual_info_dict = dict(zip(actual_io_config[f"{io_key}_names"], actual_io_config[f"{io_key}_{info_key}"]))
            for name, value in expected_info_dict.items():
                assert actual_info_dict[name] == value


def test_compose_onnx_models_llm_pipeline(tmp_path):
    # setup
    pytorch_model = make_local_tiny_llama(tmp_path)
    onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
        pytorch_model, tmp_path / "onnx_model"
    )
    split_model = create_pass_from_dict(
        SplitModel,
        {
            "split_assignments": {
                "model.embed_tokens": 0,
                "model.layers.0": 1,
                "model.layers.1": 2,
                "lm_head": 3,
            }
        },
        disable_search=True,
    ).run(onnx_model, tmp_path / "split_model")
    llm_model = create_pass_from_dict(StaticLLM, {"batch_size": 1, "context_length": 64}, disable_search=True).run(
        split_model, tmp_path / "llm_model"
    )
    # change genai config so that the context_0 model has session options
    # will check if it got carried over
    llm_genai_config_path = tmp_path / "llm_model" / "genai_config.json"
    with llm_genai_config_path.open() as f:
        llm_genai_config = json.load(f)
    session_options = {
        "provider_options": [{"qnn": {"backend_path": "QnnHtp.dll"}}],
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
    }
    llm_genai_config["model"]["decoder"]["pipeline"][0]["context_0"]["session_options"] = session_options
    with llm_genai_config_path.open("w") as f:
        json.dump(llm_genai_config, f, indent=4)

    p = create_pass_from_dict(ComposeOnnxModels, {}, disable_search=True)

    # execute
    output_model_path = tmp_path / "output_model"
    output_model = p.run(llm_model, output_model_path)

    # check
    assert isinstance(output_model, CompositeModelHandler)
    assert output_model.model_attributes["llm_pipeline"] == {
        "embeddings": "embeddings",
        "context": ["context"],
        "iterator": ["iterator"],
        "lm_head": "lm_head",
    }
    assert len(list(output_model.model_components)) == 4
    with (output_model_path / "genai_config.json").open() as f:
        genai_config = json.load(f)
    assert genai_config["model"]["decoder"]["pipeline"][0]["context"]["session_options"] == session_options
    assert genai_config["model"]["decoder"]["pipeline"][0]["iterator"]["session_options"] == session_options


def _make_component(path, input_name, output_name, mid_name, mid_op="Identity"):
    """Build a two node component: input -> mid_op -> mid_name -> Identity -> output.

    ``mid_op`` is either "Identity" or "EPContext" so that a test can choose whether ``mid_name``
    is an ordinary intermediate value or one that belongs to a compiled context binary.

    Node names are derived from the file name so that two components can collide on a *value* name
    without also colliding on a *node* name. Those are separate failure modes and separate code
    paths, and a test that triggers both cannot say which one it is covering.
    """
    tag = Path(path).stem
    tensor_type = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [2, 4])
    out_type = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [2, 4])

    if mid_op == "EPContext":
        # The name is read back as a file path and the file is copied next to the composed model,
        # so it has to exist. One per component, since the names have to stay distinct.
        cache_name = f"{tag}.bin"
        (Path(path).parent / cache_name).write_bytes(b"not a real context binary")
        first = helper.make_node(
            "EPContext",
            [input_name],
            [mid_name],
            name=f"{tag}_ctx",
            domain="com.microsoft",
            embed_mode=0,
            ep_cache_context=cache_name,
            source="QNN",
        )
    else:
        first = helper.make_node("Identity", [input_name], [mid_name], name=f"{tag}_first")

    second = helper.make_node("Identity", [mid_name], [output_name], name=f"{tag}_second")

    graph = helper.make_graph([first, second], "component", [tensor_type], [out_type])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.save(model, path)
    return path


def _compose(paths, output_path):
    return ComposeOnnxModels._get_composed_model(paths, output_path, external_config={})


def test_compose_renames_colliding_intermediate_values(tmp_path):
    """Two components that independently produced a value called "val_0" must both survive.

    Execution providers number the values they invent per compilation, so separately compiled
    components restart the same counter and collide on names that refer to different tensors.
    """
    # setup
    first = _make_component(tmp_path / "first.onnx", "x", "hidden", "val_0")
    second = _make_component(tmp_path / "second.onnx", "hidden", "y", "val_0")

    # execute
    composed = _compose([first, second], tmp_path / "composed.onnx")

    # check
    model = onnx.load(composed.model_path)
    produced = [name for node in model.graph.node for name in node.output]
    assert len(produced) == len(set(produced)), f"duplicate value names in composed graph: {produced}"
    assert "val_0" in produced, "the first component's value should keep its name"
    assert sum(name.startswith("val_0_composed") for name in produced) == 1, (
        "the second component's colliding value should have been renamed exactly once"
    )
    # the rename must be wired through: nothing may read a name that nobody produces
    available = set(produced) | {inp.name for inp in model.graph.input}
    for node in model.graph.node:
        for name in node.input:
            assert name in available, f"{node.name} reads dangling value {name}"


@pytest.mark.parametrize("reserved_by", ["input", "initializer", "intermediate", "output", "later_component"])
def test_compose_reserves_original_value_names_before_renaming(tmp_path, reserved_by):
    first = _make_component(tmp_path / "first.onnx", "x", "hidden", "val_0")
    second = _make_component(tmp_path / "second.onnx", "hidden", "y", "val_0")
    reserved = "val_0_composed1"
    paths = [first, second]
    model = onnx.load(second)
    if reserved_by == "input":
        model.graph.input.append(helper.make_tensor_value_info(reserved, TensorProto.FLOAT, [2, 4]))
    elif reserved_by == "initializer":
        model.graph.initializer.append(helper.make_tensor(reserved, TensorProto.FLOAT, [2, 4], [1.0] * 8))
    elif reserved_by == "intermediate":
        model.graph.node.append(helper.make_node("Identity", ["hidden"], [reserved], name="reserved"))
    elif reserved_by == "output":
        model.graph.node[-1].output[0] = reserved
        model.graph.output[0].name = reserved
    else:
        paths.append(_make_component(tmp_path / "third.onnx", "y", "z", reserved))
    onnx.save(model, second)

    composed = _compose(paths, tmp_path / "composed.onnx")
    model = onnx.load(composed.model_path)
    onnx.checker.check_model(model)
    assert model.graph.node[2].output[0] == "val_0_composed1_1"
    value_names = {value.name for value in [*model.graph.input, *model.graph.initializer, *model.graph.output]} | {
        name for node in model.graph.node for name in node.output
    }
    assert reserved in value_names


@pytest.mark.parametrize("with_consumer", [False, True])
def test_compose_preserves_overridable_initializer(tmp_path, with_consumer):
    path = _make_component_with_initializer(tmp_path / "first.onnx", "x", "hidden", "w")
    model = onnx.load(path)
    model.graph.input.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 4]))
    onnx.save(model, path)
    paths = [path]
    if with_consumer:
        paths.append(_make_component(tmp_path / "second.onnx", "hidden", "y", "second_mid"))
    composed = _compose(paths, tmp_path / "composed.onnx")
    model = onnx.load(composed.model_path)
    onnx.checker.check_model(model)
    assert "w" in {value.name for value in model.graph.input}
    assert "w" in {value.name for value in model.graph.initializer}
    session = onnxruntime.InferenceSession(composed.model_path, providers=["CPUExecutionProvider"])
    x = np.full((2, 4), 2, dtype=np.float32)
    np.testing.assert_array_equal(session.run(None, {"x": x})[0], x + 1)
    np.testing.assert_array_equal(session.run(None, {"x": x, "w": x + 3})[0], x + 5)


@pytest.mark.parametrize("overridable_component", [0, 1])
def test_compose_does_not_merge_fixed_and_overridable_initializers(tmp_path, overridable_component):
    first = _make_component_with_initializer(tmp_path / "first.onnx", "x", "hidden", "w")
    second = _make_component_with_initializer(tmp_path / "second.onnx", "hidden", "y", "w")
    paths = [first, second]
    model = onnx.load(paths[overridable_component])
    model.graph.input.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 4]))
    onnx.save(model, paths[overridable_component])
    composed = _compose(paths, tmp_path / "composed.onnx")
    model = onnx.load(composed.model_path)
    onnx.checker.check_model(model)
    assert len(model.graph.initializer) == 2
    override_name = next(value.name for value in model.graph.input if value.name != "x")
    session = onnxruntime.InferenceSession(composed.model_path, providers=["CPUExecutionProvider"])
    x = np.full((2, 4), 2, dtype=np.float32)
    np.testing.assert_array_equal(session.run(None, {"x": x})[0], x + 2)
    np.testing.assert_array_equal(session.run(None, {"x": x, override_name: x + 3})[0], x + 6)


def test_compose_rejects_unrenamable_epcontext_collision(tmp_path):
    """A collision on a name an EPContext node owns cannot be resolved, so it must fail loudly.

    Silently renaming it would produce a model that only fails later, at session creation, with an
    error that points nowhere near the cause.
    """
    # setup: the plain component claims "val_0" first, so the EPContext node cannot keep it
    first = _make_component(tmp_path / "first.onnx", "x", "hidden", "val_0")
    second = _make_component(tmp_path / "second.onnx", "hidden", "y", "val_0", mid_op="EPContext")

    # execute and check
    with pytest.raises(ValueError, match=r"val_0.*EPContext|EPContext.*val_0"):
        _compose([first, second], tmp_path / "composed.onnx")


def _make_component_with_initializer(path, input_name, output_name, init_name):
    """Build a one node component whose only constant is called ``init_name``.

    ``input_name`` is added to that constant, so a test can read the composed graph and tell which
    value the node actually consumes.
    """
    tag = Path(path).stem
    in_type = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [2, 4])
    out_type = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [2, 4])
    init = helper.make_tensor(init_name, TensorProto.FLOAT, [2, 4], [1.0] * 8)
    node = helper.make_node("Add", [input_name, init_name], [output_name], name=f"{tag}_add")
    graph = helper.make_graph([node], "component", [in_type], [out_type], initializer=[init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


def test_compose_renames_initializer_colliding_with_produced_value(tmp_path):
    """A component's constant must not take a name an earlier component's node already produces.

    ONNX values share one namespace, so reusing the name attaches the constant to the earlier
    component's intermediate and silently feeds this component a different tensor than it was
    compiled against.
    """
    # setup: the first component produces an intermediate "w", the second has a constant "w"
    first = _make_component(tmp_path / "first.onnx", "x", "hidden", "w")
    second = _make_component_with_initializer(tmp_path / "second.onnx", "hidden", "y", "w")

    # execute
    composed = _compose([first, second], tmp_path / "composed.onnx")

    # check
    model = onnx.load(composed.model_path)
    produced = {name for node in model.graph.node for name in node.output}
    init_names = [init.name for init in model.graph.initializer]
    assert not produced & set(init_names), f"a value is both produced and constant: {produced & set(init_names)}"
    assert "w" in produced, "the first component's intermediate should keep its name"
    renamed = [name for name in init_names if name.startswith("w_composed")]
    assert len(renamed) == 1, f"the colliding initializer should have been renamed once: {init_names}"
    add = next(node for node in model.graph.node if node.op_type == "Add")
    assert renamed[0] in add.input, f"the Add must read the renamed constant, got {list(add.input)}"


def _make_kv_sharing_components(tmp_path):
    """Build two components wired the way a KV sharing decoder splits.

    The first produces a cache tensor and a hidden state. The second consumes both, which is
    what makes the cache tensor look like an ordinary internal connection even though the
    runtime has to read it back.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    cache = helper.make_tensor_value_info("present.13.key", TensorProto.FLOAT, [2, 4])
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])

    first = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["present.13.key"], name="first_cache"),
            helper.make_node("Identity", ["x"], ["hidden"], name="first_hidden"),
        ],
        "first",
        [x],
        [cache, hidden],
    )
    second = helper.make_graph(
        [helper.make_node("Add", ["hidden", "present.13.key"], ["y"], name="second_add")],
        "second",
        [hidden, cache],
        [y],
    )

    paths = []
    for graph in (first, second):
        path = tmp_path / f"{graph.name}.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        paths.append(path)
    return paths


def test_compose_keeps_state_outputs_consumed_by_later_components(tmp_path):
    """A KV cache output that a later component reads must stay a graph output.

    Gemma 4 shares layers 13 and 14 with layers 15 through 34, so after splitting, one chunk
    produces ``present.13.key`` and later chunks consume it. Treating that as an internal
    connection drops it from the composed model, the runtime can then never refresh the slot,
    and the model emits a correct first token followed by garbage.
    """
    # setup
    paths = _make_kv_sharing_components(tmp_path)
    patterns = [re.compile(r"^present\.\d+\.key$")]

    # execute
    without = ComposeOnnxModels._get_composed_model(paths, tmp_path / "without.onnx", external_config={})
    with_state = ComposeOnnxModels._get_composed_model(
        paths, tmp_path / "with.onnx", external_config={}, state_output_patterns=patterns
    )

    # check
    dropped = {out.name for out in onnx.load(without.model_path).graph.output}
    assert dropped == {"y"}, "baseline behavior changed: only unconsumed outputs should survive"

    kept = {out.name for out in onnx.load(with_state.model_path).graph.output}
    assert kept == {"y", "present.13.key"}, "the shared cache output must be preserved"


def test_state_output_patterns_read_from_genai_config(tmp_path):
    """The patterns come from the decoder contract, not from guessing at names."""
    # setup
    genai_config = tmp_path / "genai_config.json"
    with genai_config.open("w") as f:
        json.dump(
            {
                "model": {
                    "decoder": {
                        "outputs": {
                            "logits": "logits",
                            "present_key_names": "present.%d.key",
                            "present_value_names": "present.%d.value",
                            "paged_present_names": "present.%d.page.%d",
                        }
                    }
                }
            },
            f,
        )
    model = CompositeModelHandler([], [], model_attributes={"additional_files": [str(genai_config)]})

    # execute
    patterns = ComposeOnnxModels._state_output_patterns(model)

    # check
    matched = {
        name
        for name in ("present.13.key", "present.0.value", "present.2.page.17", "logits", "hidden")
        if any(p.match(name) for p in patterns)
    }
    assert matched == {"present.13.key", "present.0.value", "present.2.page.17", "logits"}


def test_state_output_patterns_absent_genai_config(tmp_path):
    """Without a decoder contract there is nothing to preserve, and nothing should break."""
    model = CompositeModelHandler([], [], model_attributes={})
    assert ComposeOnnxModels._state_output_patterns(model) == []


@pytest.mark.parametrize("source", ["genai", "explicit"])
def test_compose_preserves_fixed_name_state_outputs(tmp_path, source):
    paths = _make_kv_sharing_components(tmp_path)
    for path in paths:
        model = onnx.load(path)
        for value in [*model.graph.input, *model.graph.output]:
            if value.name == "present.13.key":
                value.name = "state"
        for node in model.graph.node:
            for names in (node.input, node.output):
                for idx, name in enumerate(names):
                    if name == "present.13.key":
                        names[idx] = "state"
        onnx.save(model, path)
    attributes = {}
    pass_config = {"save_as_external_data": False}
    if source == "genai":
        config_path = tmp_path / "genai_config.json"
        config_path.write_text(json.dumps({"model": {"decoder": {"outputs": {"state": "state"}}}}))
        attributes["additional_files"] = [str(config_path)]
    else:
        pass_config["preserved_outputs"] = ["state"]
    model = CompositeModelHandler(
        [ONNXModelHandler(path) for path in paths], ["first", "second"], model_attributes=attributes
    )
    output = create_pass_from_dict(ComposeOnnxModels, pass_config, disable_search=True).run(
        model, tmp_path / "composed"
    )
    assert {value.name for value in onnx.load(output.model_path).graph.output} == {"state", "y"}


def test_compose_preserves_explicit_state_output_templates(tmp_path):
    paths = _make_kv_sharing_components(tmp_path)
    model = CompositeModelHandler([ONNXModelHandler(path) for path in paths], ["first", "second"])
    output = create_pass_from_dict(
        ComposeOnnxModels, {"preserved_outputs": ["present.%d.key"]}, disable_search=True
    ).run(model, tmp_path / "composed")
    assert {value.name for value in onnx.load(output.model_path).graph.output} == {"present.13.key", "y"}


def _make_pipeline_composite(tmp_path, genai_model):
    """Build the smallest composite that reaches the embedding stage branch.

    ``llm_pipeline`` deliberately has no "embeddings" entry: that is the shape produced when the
    embedding lookup stays in its own artifact outside the optimization pipeline, which is the
    only case where the genai config has to declare an embedding stage.
    """
    _make_component(tmp_path / "context.onnx", "inputs_embeds", "hidden", "c0")
    _make_component(tmp_path / "iterator.onnx", "inputs_embeds", "hidden", "i0")
    _make_component(tmp_path / "lm_head.onnx", "hidden", "logits", "l0")

    genai_config_path = tmp_path / "genai_config.json"
    with genai_config_path.open("w") as f:
        json.dump({"model": genai_model}, f)

    components = [
        ONNXModelHandler(model_path=tmp_path, onnx_file_name=f"{name}.onnx")
        for name in ("context", "iterator", "lm_head")
    ]
    return CompositeModelHandler(
        components,
        ["context", "iterator", "lm_head"],
        model_path=tmp_path,
        model_attributes={
            "llm_pipeline": {"context": ["context"], "iterator": ["iterator"], "lm_head": "lm_head"},
            "additional_files": [str(genai_config_path)],
        },
    )


def _embedding_section():
    return {
        "filename": "embedding/model.onnx",
        "inputs": {
            "input_ids": "input_ids",
            "image_features": "image_features",
            "audio_features": "audio_features",
        },
        "outputs": {"inputs_embeds": "inputs_embeds", "per_layer_inputs": "per_layer_inputs"},
    }


def test_genai_config_keeps_model_type_and_feature_inputs_for_multimodal(tmp_path):
    """A model that declares an encoder keeps its model type and every declared embedding input.

    The type selects the ort-genai model class, and only the multimodal class binds image and
    audio features. The stage input list is matched by name in decoder_only_pipeline.cpp, so a
    declared input left out of it is dropped silently rather than raising.
    """
    # setup
    model = _make_pipeline_composite(
        tmp_path,
        {
            "type": "gemma4",
            "decoder": {"filename": "decoder.onnx"},
            "embedding": _embedding_section(),
            "vision": {"filename": "vision_encoder/model.onnx"},
        },
    )

    # execute
    update_llm_pipeline_genai_config(model)

    # check
    with (tmp_path / "genai_config.json").open() as f:
        config = json.load(f)
    assert config["model"]["type"] == "gemma4", "the multimodal model class must not be overwritten"
    stage = config["model"]["decoder"]["pipeline"][0]["embedding"]
    assert stage["inputs"] == ["input_ids", "image_features", "audio_features"]


def test_genai_config_text_only_is_unchanged(tmp_path):
    """Without an encoder the generic pipeline type and the lone input_ids stage still apply."""
    # setup
    embedding = _embedding_section()
    embedding["inputs"] = {"input_ids": "input_ids"}
    model = _make_pipeline_composite(
        tmp_path,
        {"type": "llama", "decoder": {"filename": "decoder.onnx"}, "embedding": embedding},
    )

    # execute
    update_llm_pipeline_genai_config(model)

    # check
    with (tmp_path / "genai_config.json").open() as f:
        config = json.load(f)
    assert config["model"]["type"] == "decoder-pipeline"
    assert config["model"]["decoder"]["pipeline"][0]["embedding"]["inputs"] == ["input_ids"]


@pytest.mark.parametrize("embedding_options", [None, {}, {"provider_options": [], "intra_op_num_threads": 2}])
def test_genai_config_preserves_embedding_session_options(tmp_path, embedding_options):
    embedding = _embedding_section()
    if embedding_options is not None:
        embedding["session_options"] = embedding_options
    decoder_options = {"provider_options": [{"qnn": {"backend_path": "QnnHtp.dll"}}]}
    model = _make_pipeline_composite(
        tmp_path,
        {
            "type": "gemma4",
            "decoder": {"filename": "decoder.onnx", "session_options": decoder_options},
            "embedding": embedding,
            "vision": {"filename": "vision.onnx"},
        },
    )
    update_llm_pipeline_genai_config(model, group_session_options=decoder_options)
    config = json.loads((tmp_path / "genai_config.json").read_text())
    pipeline = config["model"]["decoder"]["pipeline"][0]
    assert pipeline["context"]["session_options"] == decoder_options
    if embedding_options is None:
        assert "session_options" not in pipeline["embedding"]
    else:
        assert pipeline["embedding"]["session_options"] == embedding_options


def test_genai_config_keeps_model_type_for_audio_output(tmp_path):
    """A speech output model keeps its model class even though it declares no input encoder.

    It names its two graphs under nested keys rather than a top level filename, so a plain
    filename check misses them and would strand the component that generates audio.
    """
    # setup
    model = _make_pipeline_composite(
        tmp_path,
        {
            "type": "lfm2_audio",
            "decoder": {"filename": "decoder.onnx"},
            "embedding": _embedding_section(),
            "audio_output": {
                "depthformer": {"filename": "depthformer.onnx"},
                "embedding": {"filename": "audio_embedding.onnx"},
            },
        },
    )

    # execute
    update_llm_pipeline_genai_config(model)

    # check
    with (tmp_path / "genai_config.json").open() as f:
        config = json.load(f)
    assert config["model"]["type"] == "lfm2_audio", "the speech output model class must not be overwritten"
