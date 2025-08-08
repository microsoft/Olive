# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import onnxruntime
import pytest

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import resave_model
from olive.passes.onnx.context_binary import EPContextBinaryGenerator
from test.utils import get_onnx_model


@pytest.mark.skipif(
    "QNNExecutionProvider" not in onnxruntime.get_available_providers(),
    reason="QNNExecutionProvider is not available in this environment",
)
@pytest.mark.parametrize("embed_context", [True, False])
def test_ep_context_binary_generator(tmp_path, embed_context):
    # setup
    model = get_onnx_model()
    accelerator_spec = AcceleratorSpec(
        accelerator_type="NPU",
        execution_provider="QNNExecutionProvider",
    )
    p = create_pass_from_dict(
        EPContextBinaryGenerator,
        {"embed_context": embed_context},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    # execute
    output_model = p.run(model, tmp_path)

    # assert
    assert isinstance(output_model, ONNXModelHandler)
    ctx_path = tmp_path / "dummy_model_ctx.onnx"
    assert output_model.model_path == str(ctx_path)
    assert ctx_path.exists()
    assert len(list(tmp_path.glob("dummy_model_ctx*.bin"))) == int(not embed_context)


@pytest.mark.skipif(
    "QNNExecutionProvider" not in onnxruntime.get_available_providers(),
    reason="QNNExecutionProvider is not available in this environment",
)
@pytest.mark.parametrize("is_llm", [True, False])
def test_ep_context_binary_generator_composite(tmp_path, is_llm):
    # setup
    model = get_onnx_model()

    # create a composite model
    parent_dir = tmp_path / "composite_model"
    parent_dir.mkdir(parents=True, exist_ok=True)
    component_names = ["component_0", "component_1", "component_2", "component_3"]
    model_attributes = None
    if is_llm:
        component_names = ["embeddings", *component_names, "lm_head"]
        with open(parent_dir / "genai_config.json", "w") as f:
            json.dump({"model": {"decoder": {}, "type": "decoder-pipeline"}}, f)

        model_attributes = {
            "llm_pipeline": {
                "embeddings": "embeddings",
                "context": ["component_0", "component_1"],
                "iterator": ["component_2", "component_3"],
                "lm_head": "lm_head",
            },
            "additional_files": [str(parent_dir / "genai_config.json")],
        }
    component_models = []
    for name in component_names:
        component_path = parent_dir / f"{name}.onnx"
        resave_model(model.model_path, str(component_path))
        component_models.append(ONNXModelHandler(component_path))
    composite_model = CompositeModelHandler(component_models, component_names, model_attributes=model_attributes)
    accelerator_spec = AcceleratorSpec(
        accelerator_type="NPU",
        execution_provider="QNNExecutionProvider",
    )
    p = create_pass_from_dict(
        EPContextBinaryGenerator,
        # weight sharing requires a qdq model
        # might also have issues in pytest environment
        {
            "weight_sharing": False,
            "provider_options": {
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
                "soc_model": "60",
            },
            "session_options": {"intra_op_num_threads": 2, "inter_op_num_threads": 1},
        },
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    # execute
    output_model_path = tmp_path / "output_model"
    output_model = p.run(composite_model, output_model_path)

    # assert
    assert isinstance(output_model, CompositeModelHandler)
    output_component_map = dict(output_model.get_model_components())
    assert len(output_component_map) == len(component_models)
    if is_llm:
        assert output_model.model_attributes["llm_pipeline"] == {
            "embeddings": "embeddings",
            "context": ["component_0_ctx", "component_1_ctx"],
            "iterator": ["component_2_ctx", "component_3_ctx"],
            "lm_head": "lm_head",
        }
        with open(output_model_path / "genai_config.json") as f:
            genai_config = json.load(f)
        assert set(genai_config["model"]["decoder"]["pipeline"][0].keys()) == set(output_model.model_component_names)
        session_options = genai_config["model"]["decoder"]["pipeline"][0]["component_0_ctx"]["session_options"]
        assert session_options["intra_op_num_threads"] == 2
        assert "qnn" in session_options["provider_options"][0]
        assert session_options["provider_options"][0]["qnn"]["htp_performance_mode"] == "burst"
        assert session_options["provider_options"][0]["qnn"]["backend_path"] == "QnnHtp.dll"
    for name in component_names:
        # print(output_component_map[name].model_path)
        is_skipped = name in ["embeddings", "lm_head"]
        expected_name = name if is_skipped else f"{name}_ctx"
        expected_model_path = output_model_path / f"{expected_name}.onnx"
        assert output_component_map[expected_name].model_path == str(expected_model_path)
        assert expected_model_path.exists()
        if not is_skipped:
            assert len(list(output_model_path.glob(f"{name}_ctx*.bin"))) == 1
