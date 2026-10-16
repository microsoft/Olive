# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import onnx
import onnxruntime
import pytest

from olive.common.ort_genai_config import ORT_GENAI_CONFIG_UPDATES_KEY
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


def _mock_get_available_providers():
    return ["QNNExecutionProvider", "CPUExecutionProvider"]


def test_single_target_populates_model_attributes(tmp_path):
    """Single-target mode should populate model_attributes."""
    from pathlib import Path
    from unittest.mock import patch

    accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

    p = create_pass_from_dict(
        EPContextBinaryGenerator,
        {
            "provider_options": {
                "soc_model": "60",
                "htp_performance_mode": "burst",
            },
        },
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    with (
        patch.object(EPContextBinaryGenerator, "_run_qnn") as mock_single,
        patch("onnxruntime.get_available_providers", _mock_get_available_providers),
    ):

        def side_effect(model, config, output_model_path):
            out_path = Path(output_model_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("dummy")
            return ONNXModelHandler(model_path=str(out_path))

        mock_single.side_effect = side_effect

        input_model = get_onnx_model()
        output_path = str(tmp_path / "output.onnx")
        result = p.run(input_model, output_path)

    assert isinstance(result, ONNXModelHandler)
    assert result.model_attributes["ep"] == "QNNExecutionProvider"
    assert result.model_attributes["device"] == "NPU"
    assert result.model_attributes["architecture"] == "60"
    assert result.model_attributes["provider_options"]["soc_model"] == "60"


@pytest.mark.parametrize(("share", "stop"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("failure", [None, "devices", "provider", "session"])
def test_context_binary_cleans_registration_on_failure(tmp_path, monkeypatch, share, stop, failure):
    registration = MagicMock()
    unregistration = MagicMock()
    devices = MagicMock(return_value=[SimpleNamespace(ep_name="QNNExecutionProvider")])
    options = MagicMock()
    provider_options = {"soc_model": "60"}
    session_options = {"log_severity_level": 1}
    output_path = tmp_path / "model_ctx.onnx"
    session = MagicMock(side_effect=lambda *args, **kwargs: output_path.write_bytes(b"mock context"))
    error = RuntimeError("context creation failed")
    if failure == "devices":
        devices.side_effect = error
    elif failure == "provider":
        options.add_provider_for_devices.side_effect = error
    elif failure == "session":
        session.side_effect = error
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", SimpleNamespace(get_library_path=lambda: "mock_qnn.dll"))
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(onnxruntime, "register_execution_provider_library", registration)
    monkeypatch.setattr(onnxruntime, "unregister_execution_provider_library", unregistration)
    monkeypatch.setattr(onnxruntime, "get_ep_devices", devices)
    monkeypatch.setattr(onnxruntime, "SessionOptions", lambda: options)
    monkeypatch.setattr(onnxruntime, "InferenceSession", session)

    def run():
        return EPContextBinaryGenerator._generate_context_binary(  # pylint: disable=protected-access
            "input.onnx",
            output_path,
            device="NPU",
            execution_provider="QNNExecutionProvider",
            provider_options=provider_options,
            session_options=session_options,
            embed_context=True,
            share_ep_contexts=share,
            stop_share_ep_contexts=stop,
        )

    if failure:
        with pytest.raises(RuntimeError, match="context creation failed"):
            run()
    else:
        assert isinstance(run(), ONNXModelHandler)
    assert provider_options == {"soc_model": "60"}
    assert session_options == {"log_severity_level": 1}
    registration.assert_called_once_with("QNNExecutionProvider", "mock_qnn.dll")
    if failure or not share or stop:
        unregistration.assert_called_once_with("QNNExecutionProvider")
    else:
        unregistration.assert_not_called()


def test_single_target_does_not_update_genai_config_by_default(tmp_path):
    from unittest.mock import patch

    config_path = tmp_path / "genai_config.json"
    original_config = {"model": {"vision": {"filename": "vision/model.onnx"}}}
    config_path.write_text(json.dumps(original_config), encoding="utf-8")
    input_model = get_onnx_model()
    input_model.model_attributes = {"additional_files": [str(config_path)]}
    accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )
    output_path = tmp_path / "output.onnx"
    resave_model(input_model.model_path, output_path)

    with (
        patch.object(
            EPContextBinaryGenerator,
            "_run_qnn",
            return_value=ONNXModelHandler(model_path=str(output_path)),
        ),
        patch("onnxruntime.get_available_providers", _mock_get_available_providers),
    ):
        result = context_pass.run(input_model, output_path)

    assert json.loads(config_path.read_text(encoding="utf-8")) == original_config
    assert ORT_GENAI_CONFIG_UPDATES_KEY not in result.model_attributes


def test_single_target_updates_selected_genai_config_role_when_enabled(tmp_path):
    from unittest.mock import patch

    config_path = tmp_path / "source" / "genai_config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "vision": {
                        "filename": "vision/model.onnx",
                        "session_options": {"existing": "value"},
                    },
                    "decoder": {"filename": "decoder/model.onnx"},
                }
            }
        ),
        encoding="utf-8",
    )
    input_model = get_onnx_model()
    input_model.model_attributes = {"additional_files": [str(config_path)]}
    accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {
            "update_genai_config": True,
            "genai_config_role": "vision",
            "provider_options": {"soc_model": "60"},
            "session_options": {"log_severity_level": 1},
        },
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )
    output_path = tmp_path / "output" / "vision_ctx.onnx"
    resave_model(input_model.model_path, output_path)

    with (
        patch.object(
            EPContextBinaryGenerator,
            "_run_qnn",
            return_value=ONNXModelHandler(model_path=str(output_path)),
        ),
        patch("onnxruntime.get_available_providers", _mock_get_available_providers),
    ):
        result = context_pass.run(input_model, output_path)

    updated_path = output_path.parent / "genai_config.json"
    updated = json.loads(updated_path.read_text(encoding="utf-8"))
    assert updated["model"]["decoder"] == {"filename": "decoder/model.onnx"}
    assert updated["model"]["vision"]["filename"] == "vision_ctx.onnx"
    backend_path = "libQnnHtp.so" if platform.system() == "Linux" else "QnnHtp.dll"
    assert updated["model"]["vision"]["session_options"] == {
        "existing": "value",
        "log_severity_level": 1,
        "provider_options": [{"qnn": {"soc_model": "60", "backend_path": backend_path}}],
    }
    assert str(updated_path) in result.model_attributes["additional_files"]
    assert result.model_attributes[ORT_GENAI_CONFIG_UPDATES_KEY] == [
        {
            "file_name": "genai_config.json",
            "json_paths": ["/model/vision"],
        }
    ]


@pytest.fixture(name="mock_openvino_ep")
def mock_openvino_ep_fixture(monkeypatch):
    options = MagicMock()
    npu = SimpleNamespace(
        ep_name="OpenVINOExecutionProvider",
        device=SimpleNamespace(type=onnxruntime.OrtHardwareDeviceType.NPU, device_id=0xB03E),
        ep_metadata={"npu_platform": "5010"},
    )
    gpu = SimpleNamespace(
        ep_name="OpenVINOExecutionProvider",
        device=SimpleNamespace(type=onnxruntime.OrtHardwareDeviceType.GPU, device_id=0xB03E),
        ep_metadata={"npu_platform": "5010"},
    )
    devices = MagicMock(return_value=[gpu, npu])
    registration, unregistration = MagicMock(), MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime_ep_openvino",
        SimpleNamespace(get_ep_name=lambda: "OpenVINOExecutionProvider", get_library_path=lambda: "mock_openvino.dll"),
    )
    monkeypatch.setattr(onnxruntime, "SessionOptions", lambda: options)
    monkeypatch.setattr(onnxruntime, "get_ep_devices", devices)
    monkeypatch.setattr(onnxruntime, "register_execution_provider_library", registration)
    monkeypatch.setattr(onnxruntime, "unregister_execution_provider_library", unregistration)
    return SimpleNamespace(
        options=options, npu=npu, devices=devices, registration=registration, unregistration=unregistration
    )


def _write_openvino_context(output_path, embed_context=False):
    node = onnx.helper.make_node(
        "EPContext",
        ["input"],
        ["output"],
        domain="com.microsoft",
        embed_mode=int(embed_context),
        ep_cache_context=b"\x80\xffcompiled context" if embed_context else b"model_ctx.bin",
    )
    graph = onnx.helper.make_graph(
        [node],
        "openvino_context",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    onnx.save(
        onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17), onnx.helper.make_opsetid("com.microsoft", 1)]
        ),
        output_path,
    )
    if not embed_context:
        (output_path.parent / "model_ctx.bin").write_bytes(b"compiled context")


@pytest.mark.parametrize("compile_flow", ["aot", "jit"])
@pytest.mark.parametrize("embed_context", [False, True])
def test_openvino_context_uses_selected_device_and_pass_options(
    tmp_path, monkeypatch, mock_openvino_ep, compile_flow, embed_context
):
    output_path = tmp_path / "output" / "model_ctx.onnx"
    compiler = MagicMock()
    compiler.compile_to_file.side_effect = lambda path: _write_openvino_context(output_path, embed_context)
    compiler_class = MagicMock(return_value=compiler)
    session = MagicMock(side_effect=lambda *args, **kwargs: _write_openvino_context(output_path, embed_context))
    monkeypatch.setattr(onnxruntime, "ModelCompiler", compiler_class)
    monkeypatch.setattr(onnxruntime, "InferenceSession", session)
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {
            "compile_flow": compile_flow,
            "embed_context": embed_context,
            "disable_cpu_fallback": True,
            "ep_registration_name": "ovep_registration.virtual",
            "ep_device_filters": {"npu_platform": "5010"},
            "output_model_name": "model_ctx.onnx",
            "provider_options": {"device_type": "NPU"},
            "session_options": {"log_severity_level": 1, "log_verbosity_level": 2, "custom": "value"},
        },
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type="npu", execution_provider="OpenVINOExecutionProvider"),
    )
    input_model = get_onnx_model(model_attributes={"source_attribute": "preserved"})
    result = context_pass.run(input_model, output_path.parent)

    assert result.model_path == str(output_path)
    assert result.get_resource("model_path") == str(output_path.parent)
    assert result.model_attributes["ep"] == "OpenVINOExecutionProvider"
    assert result.model_attributes["device"] == "NPU"
    assert result.model_attributes["source_attribute"] == "preserved"
    assert result.model_attributes["ep_device_filters"] == {"npu_platform": "5010"}
    assert "architecture" not in result.model_attributes
    options = mock_openvino_ep.options
    options.add_provider_for_devices.assert_called_once_with([mock_openvino_ep.npu], {"device_type": "NPU"})
    options.add_session_config_entry.assert_any_call("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry.assert_any_call("custom", "value")
    assert options.log_severity_level == 1
    assert options.log_verbosity_level == 2
    if compile_flow == "aot":
        assert not any(
            call.args[0] in {"ep.context_enable", "ep.context_embed_mode", "ep.context_file_path"}
            for call in options.add_session_config_entry.call_args_list
        )
        compiler_class.assert_called_once_with(
            options, input_model.model_path, embed_compiled_data_into_model=embed_context
        )
        compiler.compile_to_file.assert_called_once_with(str(output_path))
        session.assert_not_called()
    else:
        options.add_session_config_entry.assert_any_call("ep.context_enable", "1")
        options.add_session_config_entry.assert_any_call("ep.context_embed_mode", str(int(embed_context)))
        options.add_session_config_entry.assert_any_call("ep.context_file_path", str(output_path))
        session.assert_called_once_with(input_model.model_path, sess_options=options)
        compiler_class.assert_not_called()
    mock_openvino_ep.registration.assert_called_once_with("ovep_registration.virtual", "mock_openvino.dll")
    mock_openvino_ep.unregistration.assert_called_once_with("ovep_registration.virtual")


@pytest.mark.parametrize("failure", ["devices", "provider", "compile"])
@pytest.mark.parametrize("already_registered", [False, True])
def test_openvino_context_preserves_registration_ownership_on_failure(
    tmp_path, monkeypatch, mock_openvino_ep, failure, already_registered
):
    if already_registered:
        mock_openvino_ep.registration.side_effect = RuntimeError("EP already registered")
    compiler = MagicMock()
    monkeypatch.setattr(onnxruntime, "ModelCompiler", MagicMock(return_value=compiler))
    error = RuntimeError("compilation failed")
    if failure == "devices":
        mock_openvino_ep.devices.side_effect = error
    elif failure == "provider":
        mock_openvino_ep.options.add_provider_for_devices.side_effect = error
    else:
        compiler.compile_to_file.side_effect = error
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type="npu", execution_provider="OpenVINOExecutionProvider"),
    )
    with pytest.raises(RuntimeError, match="compilation failed"):
        context_pass.run(get_onnx_model(), tmp_path / "output")
    if already_registered:
        mock_openvino_ep.unregistration.assert_not_called()
    else:
        mock_openvino_ep.unregistration.assert_called_once_with("OpenVINOExecutionProvider")


@pytest.mark.parametrize(
    ("artifact", "error_type", "message"),
    [
        ("missing_model", FileNotFoundError, "EPContext model was not generated"),
        ("no_context", RuntimeError, "No EPContext nodes"),
        ("missing_binary", FileNotFoundError, "EPContext binary"),
    ],
)
def test_openvino_context_rejects_incomplete_output(
    tmp_path, monkeypatch, mock_openvino_ep, artifact, error_type, message
):
    output_path = tmp_path / "output" / "model_ctx.onnx"

    def compile_to_file(path):
        if artifact == "no_context":
            onnx.save(onnx.load(get_onnx_model().model_path), output_path)
        elif artifact == "missing_binary":
            _write_openvino_context(output_path)
            (output_path.parent / "model_ctx.bin").unlink()

    compiler = MagicMock()
    compiler.compile_to_file.side_effect = compile_to_file
    monkeypatch.setattr(onnxruntime, "ModelCompiler", MagicMock(return_value=compiler))
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {"output_model_name": "model_ctx.onnx"},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type="npu", execution_provider="OpenVINOExecutionProvider"),
    )
    with pytest.raises(error_type, match=message):
        context_pass.run(get_onnx_model(), output_path.parent)
    assert not output_path.exists()
    assert not list(output_path.parent.glob("model_ctx*.bin"))


@pytest.mark.parametrize(
    ("device_id", "expected"),
    [
        ("B03E", True),
        ("b03e", True),
        ("0xB03E", True),
        ("0Xb03e", True),
        ("00B03E", True),
        (0xB03E, True),
        ("B03F", False),
    ],
)
def test_openvino_device_id_matches_numeric_hex_value(mock_openvino_ep, device_id, expected):
    assert (
        EPContextBinaryGenerator._ep_device_matches(  # pylint: disable=protected-access
            mock_openvino_ep.npu, {"device_id": device_id}
        )
        is expected
    )


@pytest.mark.parametrize("compile_flow", ["aot", "jit"])
@pytest.mark.parametrize("failure", ["compile", "validation"])
def test_openvino_context_retries_same_output_after_failure(
    tmp_path, monkeypatch, mock_openvino_ep, compile_flow, failure
):
    output_path = tmp_path / "output" / "model_ctx.onnx"
    output_path.parent.mkdir()
    unrelated_path = output_path.parent / "unrelated.bin"
    unrelated_path.write_bytes(b"existing unrelated data")
    input_model = get_onnx_model()

    def compile_to_file(path):
        _write_openvino_context(output_path)
        if writer.call_count == 1:
            if failure == "compile":
                raise RuntimeError("transient compilation failure")
            onnx.save(onnx.load(input_model.model_path), output_path)

    writer = MagicMock(side_effect=compile_to_file)
    compiler = MagicMock()
    compiler.compile_to_file = writer
    monkeypatch.setattr(onnxruntime, "ModelCompiler", MagicMock(return_value=compiler))
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *args, **kwargs: writer(str(output_path)))
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {"compile_flow": compile_flow, "output_model_name": "model_ctx.onnx"},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type="npu", execution_provider="OpenVINOExecutionProvider"),
    )
    with pytest.raises(RuntimeError, match=r"transient compilation failure|No EPContext nodes"):
        context_pass.run(input_model, output_path.parent)

    assert not output_path.exists()
    assert not (output_path.parent / "model_ctx.bin").exists()
    assert unrelated_path.read_bytes() == b"existing unrelated data"

    result = context_pass.run(input_model, output_path.parent)

    assert result.model_path == str(output_path)
    assert writer.call_count == 2
    assert unrelated_path.read_bytes() == b"existing unrelated data"


@pytest.mark.parametrize("existing_output", ["model_ctx.onnx", "model_ctx.bin"])
def test_openvino_context_preserves_existing_output(tmp_path, monkeypatch, mock_openvino_ep, existing_output):
    existing_path = tmp_path / existing_output
    existing_path.write_bytes(b"existing context output")
    compiler_class = MagicMock()
    monkeypatch.setattr(onnxruntime, "ModelCompiler", compiler_class)
    context_pass = create_pass_from_dict(
        EPContextBinaryGenerator,
        {"output_model_name": "model_ctx.onnx"},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type="npu", execution_provider="OpenVINOExecutionProvider"),
    )
    with pytest.raises(FileExistsError, match="EPContext output already exists"):
        context_pass.run(get_onnx_model(), tmp_path)
    assert existing_path.read_bytes() == b"existing context output"
    compiler_class.assert_not_called()


def test_openvino_workflow_discovers_plugin_before_virtual_target_validation(tmp_path, monkeypatch, mock_openvino_ep):
    from pathlib import Path

    from olive.workflows.run.run import run

    registered = set()
    builtin_cpu = SimpleNamespace(ep_name="CPUExecutionProvider")
    openvino_cpu = SimpleNamespace(
        ep_name="OpenVINOExecutionProvider",
        device=SimpleNamespace(type=onnxruntime.OrtHardwareDeviceType.CPU, device_id=0),
        ep_metadata={},
    )

    def register(name, library_path):
        if name in registered:
            raise RuntimeError("EP already registered")
        registered.add(name)

    def devices():
        result = [builtin_cpu]
        if "OpenVINOExecutionProvider" in registered:
            result.append(openvino_cpu)
        if "ovep_registration.virtual" in registered:
            result.append(mock_openvino_ep.npu)
        return result

    monkeypatch.setattr("olive.common.ort_inference.ort_supports_ep_devices", lambda: True)
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(onnxruntime, "get_all_providers", lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    mock_openvino_ep.registration.side_effect = register
    mock_openvino_ep.unregistration.side_effect = registered.remove
    mock_openvino_ep.devices.side_effect = devices
    compiler = MagicMock()
    compiler.compile_to_file.side_effect = lambda path: _write_openvino_context(Path(path))
    monkeypatch.setattr(onnxruntime, "ModelCompiler", MagicMock(return_value=compiler))
    cache_dir = tmp_path / "cache"

    run(
        {
            "input_model": get_onnx_model().to_json(),
            "systems": {
                "local": {
                    "type": "LocalSystem",
                    "accelerators": [{"device": "npu", "execution_providers": ["OpenVINOExecutionProvider"]}],
                }
            },
            "host": "local",
            "target": "local",
            "passes": {
                "context": {
                    "type": "EPContextBinaryGenerator",
                    "ep_registration_name": "ovep_registration.virtual",
                    "ep_device_filters": {"npu_platform": "5010"},
                    "output_model_name": "model_ctx.onnx",
                }
            },
            "engine": {
                "cache_dir": str(cache_dir),
                "output_dir": str(tmp_path / "output"),
                "search_strategy": False,
                "evaluate_input_model": False,
                "no_artifacts": True,
                "log_to_file": False,
            },
        }
    )

    compiler.compile_to_file.assert_called_once()
    assert list(cache_dir.rglob("model_ctx.onnx"))
    assert registered == {"OpenVINOExecutionProvider"}
    mock_openvino_ep.registration.assert_any_call("OpenVINOExecutionProvider", "mock_openvino.dll")
    mock_openvino_ep.registration.assert_any_call("ovep_registration.virtual", "mock_openvino.dll")
    mock_openvino_ep.unregistration.assert_called_once_with("ovep_registration.virtual")


def test_openvino_model_compiler_options_parse_json_values():
    options = {
        "embed_compiled_data_into_model": "false",
        "external_initializers_size_threshold": "32",
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "flags": "NONE|ERROR_IF_NO_NODES_COMPILED",
    }
    result = EPContextBinaryGenerator._build_model_compiler_kwargs(onnxruntime, options)  # pylint: disable=protected-access
    assert result["embed_compiled_data_into_model"] is False
    assert result["external_initializers_size_threshold"] == 32
    assert result["graph_optimization_level"] == onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert result["flags"] == onnxruntime.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED
    assert options["embed_compiled_data_into_model"] == "false"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"invalid_option": True}, "Unsupported model_compiler_options"),
        ({"flags": "UNKNOWN"}, "Invalid flags member"),
        ({"embed_compiled_data_into_model": "not-a-bool"}, "valid boolean"),
    ],
)
def test_openvino_model_compiler_options_reject_invalid_values(options, message):
    with pytest.raises(ValueError, match=message):
        EPContextBinaryGenerator._build_model_compiler_kwargs(onnxruntime, options)  # pylint: disable=protected-access
