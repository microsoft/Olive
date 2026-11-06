# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
import os
import platform
import sys
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import onnx
import onnxruntime
import pytest
from onnx import helper

from olive.common.ort_genai_config import ORT_GENAI_CONFIG_UPDATES_KEY
from olive.common.ort_inference import get_ort_hardware_device_type
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import resave_model
from olive.passes.onnx.context_binary import EPContextBinaryGenerator, registered_ep_library
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
        patch.object(EPContextBinaryGenerator, "_run_single_target") as mock_single,
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
        return EPContextBinaryGenerator._generate_context_binary(
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
            "_run_single_target",
            return_value=ONNXModelHandler(model_path=str(output_path)),
        ),
        patch("onnxruntime.get_available_providers", _mock_get_available_providers),
    ):
        result = context_pass.run(input_model, output_path)

    assert json.loads(config_path.read_text(encoding="utf-8")) == original_config
    assert ORT_GENAI_CONFIG_UPDATES_KEY not in result.model_attributes


def test_single_target_updates_selected_genai_config_role_when_enabled(tmp_path):
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
            "_run_single_target",
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


def _openvino_pass(config):
    accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="OpenVINOExecutionProvider")
    return create_pass_from_dict(
        EPContextBinaryGenerator, config, disable_search=True, accelerator_spec=accelerator_spec
    )


def test_openvino_single_target_applies_epcontext_options(tmp_path):
    source = get_onnx_model()
    p = _openvino_pass(
        {
            "embed_context": True,
            "disable_cpu_fallback": True,
            "compile_flow": "aot",
            "model_compiler_options": {"graph_optimization_level": "ORT_DISABLE_ALL"},
        }
    )

    def generate(**kwargs):
        output_path = Path(kwargs["output_model_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("ctx")
        return ONNXModelHandler(model_path=str(output_path.parent), onnx_file_name=output_path.name)

    with patch.object(EPContextBinaryGenerator, "_generate_plugin_ep_context_binary", side_effect=generate) as mock_gen:
        result = p.run(source, str(tmp_path / "out"))

    compile_kwargs = mock_gen.call_args.kwargs
    assert compile_kwargs["session_options"] == {
        "ep.context_enable": "1",
        "ep.context_embed_mode": "1",
        "session.disable_cpu_ep_fallback": "1",
    }
    assert compile_kwargs["model_compiler_options"] == {
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "embed_compiled_data_into_model": True,
    }
    assert result.model_attributes["ep"] == "OpenVINOExecutionProvider"
    assert result.model_attributes["source_model_path"] == str(source.model_path)


@pytest.mark.parametrize("option", [{"update_genai_config": True}, {"genai_config_role": "decoder"}])
def test_openvino_rejects_genai_config_options(tmp_path, option):
    p = _openvino_pass(option)
    with pytest.raises(ValueError, match="only supported for QNNExecutionProvider"):
        p.run(get_onnx_model(), str(tmp_path / "out"))


def test_apply_session_options_sets_typed_attributes_and_config_entries():
    sess_options = MagicMock()
    EPContextBinaryGenerator._apply_session_options(
        sess_options, {"log_severity_level": "3", "log_verbosity_level": 2, "ep.context_enable": 1}
    )

    assert sess_options.log_severity_level == 3
    assert sess_options.log_verbosity_level == 2
    sess_options.add_session_config_entry.assert_called_once_with("ep.context_enable", "1")


def _openvino_compile_options(config):
    return EPContextBinaryGenerator._get_openvino_compile_options(_openvino_pass(config).config)


def test_openvino_explicit_aot_embed_option_is_not_overridden_by_default_embed_context():
    session_options, model_compiler_options = _openvino_compile_options(
        {"compile_flow": "aot", "model_compiler_options": {"embed_compiled_data_into_model": True}}
    )

    assert model_compiler_options["embed_compiled_data_into_model"] is True
    assert session_options["ep.context_embed_mode"] == "1"


def test_openvino_explicit_session_embed_mode_is_not_overridden_by_default_embed_context():
    session_options, model_compiler_options = _openvino_compile_options(
        {"compile_flow": "jit", "session_options": {"ep.context_embed_mode": "1"}}
    )

    assert session_options["ep.context_embed_mode"] == "1"
    assert "embed_compiled_data_into_model" not in model_compiler_options


def test_openvino_explicit_disable_cpu_fallback_is_not_overridden_by_default_flag():
    session_options, _ = _openvino_compile_options({"session_options": {"session.disable_cpu_ep_fallback": 1}})

    assert session_options["session.disable_cpu_ep_fallback"] == "1"


def test_openvino_embed_context_flag_wins_over_explicit_off_and_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        session_options, model_compiler_options = _openvino_compile_options(
            {
                "compile_flow": "aot",
                "embed_context": True,
                "session_options": {"ep.context_embed_mode": "0"},
                "model_compiler_options": {"embed_compiled_data_into_model": False},
            }
        )

    assert session_options["ep.context_embed_mode"] == "1"
    assert model_compiler_options["embed_compiled_data_into_model"] is True
    assert "embed_context is enabled" in caplog.text


def test_openvino_disable_cpu_fallback_flag_wins_over_explicit_off():
    session_options, _ = _openvino_compile_options(
        {"disable_cpu_fallback": True, "session_options": {"session.disable_cpu_ep_fallback": "0"}}
    )

    assert session_options["session.disable_cpu_ep_fallback"] == "1"


def test_openvino_embed_is_on_when_any_explicit_option_is_on():
    session_options, model_compiler_options = _openvino_compile_options(
        {
            "compile_flow": "aot",
            "session_options": {"ep.context_embed_mode": "0"},
            "model_compiler_options": {"embed_compiled_data_into_model": True},
        }
    )

    assert session_options["ep.context_embed_mode"] == "1"
    assert model_compiler_options["embed_compiled_data_into_model"] is True


def test_openvino_rejects_non_boolean_epcontext_option_value():
    with pytest.raises(ValueError, match="must be a boolean or '0'/'1'"):
        _openvino_compile_options({"session_options": {"ep.context_embed_mode": "maybe"}})


def test_openvino_jit_ignores_aot_only_embed_compiler_option():
    # model_compiler_options only applies to the aot flow, so it must not conflict with the jit embed mode
    session_options, _ = _openvino_compile_options(
        {
            "compile_flow": "jit",
            "embed_context": True,
            "model_compiler_options": {"embed_compiled_data_into_model": False},
        }
    )

    assert session_options["ep.context_embed_mode"] == "1"


def test_model_compiler_options_rejects_unknown_enum_member():
    fake_enum = type("FakeEnum", (), {"__members__": {"VALID": object()}})
    default_flags = fake_enum()

    class FakeModelCompiler:
        def __init__(self, flags=default_flags):
            pass

    class FakeOrt:
        ModelCompiler = FakeModelCompiler

    with pytest.raises(ValueError, match=r"Unsupported FakeEnum member 'TYPO'.*\['VALID'\]"):
        EPContextBinaryGenerator._build_model_compiler_kwargs(FakeOrt, {"flags": "TYPO"})


def test_openvino_rejects_weight_sharing(tmp_path):
    p = _openvino_pass({"weight_sharing": True})

    with pytest.raises(ValueError, match="weight_sharing is not supported for OpenVINO"):
        p.run(get_onnx_model(), str(tmp_path / "out"))


def test_openvino_target_devices_returns_composite_with_component_per_device(tmp_path):
    source = get_onnx_model()
    p = _openvino_pass(
        {
            "target_devices": ["643e", "B03E"],
            "compile_flow": "aot",
            "embed_context": True,
            "disable_cpu_fallback": True,
        }
    )
    filters = {"643E": {"npu_platform": "4000"}, "B03E": {"device_id": "B03E"}}

    def generate(**kwargs):
        output_path = Path(kwargs["output_model_path"])
        output_path.parent.mkdir(parents=True)
        output_path.write_text("ctx")
        return ONNXModelHandler(model_path=str(output_path.parent), onnx_file_name=output_path.name)

    with (
        patch.object(EPContextBinaryGenerator, "_resolve_openvino_device_filters", return_value=filters) as resolve,
        patch.object(EPContextBinaryGenerator, "_generate_plugin_ep_context_binary", side_effect=generate) as mock_gen,
    ):
        result = p.run(source, str(tmp_path / "out"))

    assert resolve.call_args.args == (None, ["643E", "B03E"], "npu")
    assert isinstance(result, CompositeModelHandler)
    assert result.model_component_names == ["643E", "B03E"]
    assert [call.kwargs["ep_device_filters"] for call in mock_gen.call_args_list] == list(filters.values())
    assert [call.kwargs["session_options"] for call in mock_gen.call_args_list] == [
        {
            "ep.context_enable": "1",
            "ep.context_embed_mode": "1",
            "session.disable_cpu_ep_fallback": "1",
        }
    ] * 2
    assert [call.kwargs["model_compiler_options"] for call in mock_gen.call_args_list] == [
        {"embed_compiled_data_into_model": True}
    ] * 2
    assert [Path(call.kwargs["output_model_path"]).parent.name for call in mock_gen.call_args_list] == ["643E", "B03E"]
    for (_, component), device_id in zip(result.get_model_components(), filters):
        assert Path(component.model_path) == tmp_path / "out" / device_id / "dummy_model_ctx.onnx"
        assert component.model_attributes["device_id"] == device_id
    assert result.model_attributes["no_flatten"] is True
    assert result.model_attributes["source_model_path"] == str(source.model_path)


def test_openvino_target_devices_conflicts_with_ep_device_filters(tmp_path):
    p = _openvino_pass({"target_devices": ["643E"], "ep_device_filters": {"npu_platform": "4000"}})
    with pytest.raises(ValueError, match="mutually exclusive"):
        p.run(get_onnx_model(), str(tmp_path / "out"))


def _ep_device(device_id, npu_platform=None, device_type="npu"):
    metadata = {"npu_platform": npu_platform} if npu_platform else {}
    return SimpleNamespace(
        ep_name="OpenVINOExecutionProvider",
        device=SimpleNamespace(device_id=device_id, type=get_ort_hardware_device_type(device_type)),
        ep_metadata=metadata,
    )


def _fake_ep_package(monkeypatch, devices):
    monkeypatch.setattr(
        EPContextBinaryGenerator,
        "_import_ep_package",
        staticmethod(
            lambda _: SimpleNamespace(get_ep_name=lambda: "OpenVINOExecutionProvider", get_library_path=lambda: "lib")
        ),
    )
    monkeypatch.setattr(onnxruntime, "register_execution_provider_library", lambda *_: None)
    monkeypatch.setattr(onnxruntime, "unregister_execution_provider_library", lambda *_: None)
    monkeypatch.setattr(onnxruntime, "get_ep_devices", lambda: devices)


def test_resolve_openvino_device_filters_prefers_npu_platform(monkeypatch):
    # a physical and a virtual device share id 0x643E; the virtual one carries npu_platform
    _fake_ep_package(monkeypatch, [_ep_device(0x643E), _ep_device(0x643E, "4000"), _ep_device(0xB03E)])

    filters = EPContextBinaryGenerator._resolve_openvino_device_filters("reg", ["643E", "B03E"], "npu")

    assert filters == {"643E": {"npu_platform": "4000"}, "B03E": {"device_id": "B03E"}}


def test_resolve_openvino_device_filters_fails_for_missing_device(monkeypatch):
    _fake_ep_package(monkeypatch, [_ep_device(0x643E, "4000")])

    with pytest.raises(ValueError, match=r"not discovered.*B03E"):
        EPContextBinaryGenerator._resolve_openvino_device_filters(None, ["643E", "B03E"], "npu")


def test_check_plugin_ep_ort_version_rejects_older_ort():
    EPContextBinaryGenerator._check_plugin_ep_ort_version("1.23.0")
    EPContextBinaryGenerator._check_plugin_ep_ort_version("1.29.1")
    with pytest.raises(RuntimeError, match=r"requires ONNX Runtime >= 1\.23\.0"):
        EPContextBinaryGenerator._check_plugin_ep_ort_version("1.22.2")


def test_plugin_ep_compile_checks_ort_version_before_importing_plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(onnxruntime, "__version__", "1.22.2")

    with pytest.raises(RuntimeError, match="requires ONNX Runtime"):
        EPContextBinaryGenerator._generate_plugin_ep_context_binary(
            ep_package_name="missing_plugin_package",
            model_path=str(tmp_path / "model.onnx"),
            output_model_path=tmp_path / "out" / "model_ctx.onnx",
            compile_flow="aot",
        )


def test_registered_ep_library_unregisters_when_owned():
    ort = MagicMock()
    with registered_ep_library(ort, "name", "lib"):
        pass
    ort.register_execution_provider_library.assert_called_once()
    ort.unregister_execution_provider_library.assert_called_once()


def test_registered_ep_library_skips_unregister_when_not_owned():
    ort = MagicMock()
    ort.register_execution_provider_library.side_effect = RuntimeError("EP library already registered")
    with registered_ep_library(ort, "name", "lib"):
        pass
    ort.unregister_execution_provider_library.assert_not_called()


def test_registered_ep_library_reuses_exposed_ep_after_registration_failure(caplog):
    ort = MagicMock()
    ort.register_execution_provider_library.side_effect = RuntimeError("registration failed")
    ort.get_ep_devices.return_value = [SimpleNamespace(ep_name="EP")]

    with (
        caplog.at_level("WARNING", logger="olive.passes.onnx.context_binary"),
        registered_ep_library(ort, "name", "lib", ep_name="EP"),
    ):
        pass

    assert "Reusing the already exposed 'EP' execution provider" in caplog.text
    ort.unregister_execution_provider_library.assert_not_called()


def test_registered_ep_library_does_not_reuse_exposed_ep_for_virtual_name():
    ort = MagicMock()
    ort.register_execution_provider_library.side_effect = RuntimeError("registration failed")
    ort.get_ep_devices.return_value = [SimpleNamespace(ep_name="EP")]

    with (
        pytest.raises(RuntimeError, match="registration failed"),
        registered_ep_library(ort, "name.virtual", "lib", ep_name="EP"),
    ):
        pass

    ort.unregister_execution_provider_library.assert_not_called()


def test_registered_ep_library_propagates_real_error():
    ort = MagicMock()
    ort.register_execution_provider_library.side_effect = RuntimeError("bad path")
    with pytest.raises(RuntimeError, match="bad path"), registered_ep_library(ort, "name", "lib"):
        pass
    ort.unregister_execution_provider_library.assert_not_called()


def test_registered_ep_library_serializes_overlapping_scopes():
    ort = MagicMock()
    attempted = Event()
    entered_second = Event()

    def second_scope():
        attempted.set()
        with registered_ep_library(ort, "name", "lib"):
            entered_second.set()

    with registered_ep_library(ort, "name", "lib"):
        thread = Thread(target=second_scope)
        thread.start()
        assert attempted.wait(timeout=2)
        assert not entered_second.wait(timeout=0.05)

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert entered_second.is_set()
    assert ort.register_execution_provider_library.call_count == 2
    assert ort.unregister_execution_provider_library.call_count == 2


@pytest.mark.parametrize("device_id", ["B03E", "b03e", "0xB03E", "0x00b03e", " B03E ", 0xB03E])
def test_normalize_device_id_accepts_equivalent_representations(device_id):
    assert EPContextBinaryGenerator._normalize_device_id(device_id) == "B03E"


@pytest.mark.parametrize("device_id", ["NPU", "B_03E", "-B03E", "", "0x", -1, True, None])
def test_normalize_device_id_rejects_non_hex_value(device_id):
    with pytest.raises(ValueError, match="Invalid device id"):
        EPContextBinaryGenerator._normalize_device_id(device_id)


def test_ep_device_matches_compares_device_id_numerically():
    ep_device = _ep_device(0xB03E)

    assert EPContextBinaryGenerator._ep_device_matches(ep_device, {"device_id": "0x00b03e"})
    assert EPContextBinaryGenerator._ep_device_matches(ep_device, {"device_id": 0xB03E})
    assert not EPContextBinaryGenerator._ep_device_matches(ep_device, {"device_id": "643E"})


def test_openvino_target_devices_normalizes_and_deduplicates_device_ids(tmp_path):
    p = _openvino_pass({"target_devices": ["0x643e", "643E", 0xB03E]})

    with (
        patch.object(
            EPContextBinaryGenerator, "_resolve_openvino_device_filters", side_effect=RuntimeError("stop")
        ) as resolve,
        pytest.raises(RuntimeError, match="stop"),
    ):
        p.run(get_onnx_model(), str(tmp_path / "out"))

    assert resolve.call_args.args == (None, ["643E", "B03E"], "npu")


def test_resolve_openvino_device_filters_ignores_devices_of_other_type(monkeypatch):
    # B03E is exposed, but as a GPU while the accelerator is an NPU
    _fake_ep_package(monkeypatch, [_ep_device(0x643E, "4000"), _ep_device(0xB03E, device_type="gpu")])

    with pytest.raises(ValueError, match=r"NPU target devices were not discovered: \['B03E'\]"):
        EPContextBinaryGenerator._resolve_openvino_device_filters(None, ["643E", "B03E"], "npu")


def test_select_ep_device_filters_by_accelerator_device_type():
    npu = _ep_device(0x643E)
    cpu = _ep_device(0x1, device_type="cpu")
    gpu = _ep_device(0x2, device_type="gpu")
    ort = SimpleNamespace(get_ep_devices=lambda: [cpu, gpu, npu])

    assert EPContextBinaryGenerator._select_ep_device(ort, "OpenVINOExecutionProvider", None, "npu") is npu


def test_select_ep_device_prefers_virtual_device_sharing_physical_device_id():
    virtual = _ep_device(0xB03E, "5010")
    ort = SimpleNamespace(get_ep_devices=lambda: [_ep_device(0xB03E), virtual])

    selected = EPContextBinaryGenerator._select_ep_device(
        ort, "OpenVINOExecutionProvider", {"device_id": "B03E"}, "npu"
    )

    assert selected is virtual


def test_select_ep_device_fails_when_ambiguous():
    ort = SimpleNamespace(get_ep_devices=lambda: [_ep_device(0x643E), _ep_device(0xB03E)])

    with pytest.raises(ValueError, match="Expected exactly one NPU OrtEpDevice"):
        EPContextBinaryGenerator._select_ep_device(ort, "OpenVINOExecutionProvider", None, "npu")


def test_check_weightless_ort_version_requires_ort_1_30():
    EPContextBinaryGenerator._check_weightless_ort_version("1.23.0", {"ep.context_enable": "1"})
    EPContextBinaryGenerator._check_weightless_ort_version("1.30.0", {"ep.enable_weightless": "1"})
    with pytest.raises(ValueError, match=r"ep\.enable_weightless_ep_context_nodes"):
        EPContextBinaryGenerator._check_weightless_ort_version("1.29.1", {"ep.enable_weightless": "1"})


def test_set_ep_context_binary_information_is_skipped_without_binding(tmp_path, caplog):
    with caplog.at_level("INFO", logger="olive.passes.onnx.context_binary"):
        EPContextBinaryGenerator._set_ep_context_binary_information(SimpleNamespace(), tmp_path, "model")

    assert "set_ep_context_binary_information is not available" in caplog.text


def test_set_ep_context_binary_information_passes_directory_with_trailing_separator(tmp_path):
    compiler = MagicMock()

    EPContextBinaryGenerator._set_ep_context_binary_information(compiler, tmp_path, "model")

    compiler.set_ep_context_binary_information.assert_called_once_with(os.path.join(tmp_path, ""), "model")


def _save_model(path, nodes, inputs=("x",), outputs=("y",)):
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1]) for name in inputs],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1]) for name in outputs],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(graph), str(path))


def _ep_context_node(embed_mode, context):
    return helper.make_node(
        "EPContext", ["x"], ["y"], domain="com.microsoft", embed_mode=embed_mode, ep_cache_context=context
    )


def test_validate_ep_context_model_accepts_existing_external_binary(tmp_path):
    model_path = tmp_path / "model_ctx.onnx"
    _save_model(model_path, [_ep_context_node(0, "model_ctx.bin")])
    (tmp_path / "model_ctx.bin").write_bytes(b"blob")

    EPContextBinaryGenerator._validate_ep_context_model(model_path)


def test_validate_ep_context_model_does_not_decode_embedded_binary(tmp_path):
    model_path = tmp_path / "model_ctx.onnx"
    _save_model(model_path, [_ep_context_node(1, b"\xff\xfe not utf-8")])

    EPContextBinaryGenerator._validate_ep_context_model(model_path)


def test_validate_ep_context_model_finds_ep_context_in_subgraph(tmp_path):
    then_graph = helper.make_graph(
        [_ep_context_node(1, b"blob")], "then", [], [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])]
    )
    if_node = helper.make_node("If", ["cond"], ["y"], then_branch=then_graph, else_branch=then_graph)
    model_path = tmp_path / "model_ctx.onnx"
    _save_model(model_path, [if_node], inputs=("cond",))

    EPContextBinaryGenerator._validate_ep_context_model(model_path)


def test_validate_ep_context_model_rejects_missing_external_binary(tmp_path):
    model_path = tmp_path / "model_ctx.onnx"
    _save_model(model_path, [_ep_context_node(0, "model_ctx.bin")])

    with pytest.raises(FileNotFoundError, match=r"model_ctx\.bin"):
        EPContextBinaryGenerator._validate_ep_context_model(model_path)


def test_validate_ep_context_model_rejects_model_without_ep_context_nodes(tmp_path):
    model_path = tmp_path / "model_ctx.onnx"
    _save_model(model_path, [helper.make_node("Relu", ["x"], ["y"])])

    with pytest.raises(RuntimeError, match="No EPContext nodes were generated"):
        EPContextBinaryGenerator._validate_ep_context_model(model_path)


def _fake_model_compiler(nodes):
    class FakeModelCompiler:
        def __init__(self, sess_options, model_path, **kwargs):
            pass

        def compile_to_file(self, output_path):
            output_path = Path(output_path)
            _save_model(output_path, nodes)
            output_path.with_suffix(".bin").write_bytes(b"blob")

    return FakeModelCompiler


def test_plugin_ep_compile_cleans_failed_output_and_allows_retry(tmp_path, monkeypatch):
    _fake_ep_package(monkeypatch, [_ep_device(0x643E)])
    monkeypatch.setattr(onnxruntime, "__version__", "1.32.0")
    monkeypatch.setattr(onnxruntime, "SessionOptions", MagicMock)
    output_path = tmp_path / "out" / "model_ctx.onnx"

    def compile_model():
        return EPContextBinaryGenerator._generate_plugin_ep_context_binary(
            ep_package_name="onnxruntime_ep_openvino",
            model_path=str(tmp_path / "model.onnx"),
            output_model_path=output_path,
            compile_flow="aot",
            device="npu",
        )

    # the EP compiled no node, so the output is not an EPContext model
    failing_compiler = _fake_model_compiler([helper.make_node("Relu", ["x"], ["y"])])
    monkeypatch.setattr(onnxruntime, "ModelCompiler", failing_compiler, raising=False)
    with pytest.raises(RuntimeError, match="No EPContext nodes were generated"):
        compile_model()
    assert not output_path.exists()
    assert not list(output_path.parent.glob("*.bin"))

    # the retry at the same output path is not blocked by the failed attempt
    working_compiler = _fake_model_compiler([_ep_context_node(0, "model_ctx.bin")])
    monkeypatch.setattr(onnxruntime, "ModelCompiler", working_compiler, raising=False)
    result = compile_model()

    assert Path(result.model_path) == output_path
    assert (output_path.parent / "model_ctx.bin").exists()
