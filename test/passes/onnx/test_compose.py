# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
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
