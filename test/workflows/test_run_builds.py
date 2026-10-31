# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import sys
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
from unittest.mock import MagicMock, patch

import pytest

from olive.common import mobius_utils
from olive.workflows import run as olive_run
from olive.workflows.run.builds import expand_builds
from test.passes.pytorch.test_quantization_utils import tied_word_embedding_group
from test.utils import get_pytorch_model_io_config, pytorch_model_loader

# pylint: disable=attribute-defined-outside-init

PT_MODEL = {
    "type": "PyTorchModel",
    "config": {
        "model_loader": pytorch_model_loader,
        "io_config": get_pytorch_model_io_config(),
    },
}


class TestRunBuilds:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.cache_dir = tmp_path / "cache"
        self.template = {
            "input_model": PT_MODEL,
            "passes": {
                "convert": {"type": "OnnxConversion"},
                "tune": {"type": "OrtSessionParamsTuning"},
            },
            "engine": {
                "evaluate_input_model": False,
                "cache_dir": str(self.cache_dir),
            },
        }

    def _patch_engine_and_acc(self):
        run_mock = MagicMock(return_value=MagicMock(name="WorkflowOutput"))
        acc_mock = MagicMock(name="accelerator_spec")
        engine_run_patch = patch("olive.engine.engine.Engine.run", run_mock)
        accelerator_patch = patch.object(sys.modules[olive_run.__module__], "create_accelerator", return_value=acc_mock)
        return run_mock, acc_mock, engine_run_patch, accelerator_patch

    @staticmethod
    def _patch_tied_components(monkeypatch):
        shared_weight = mobius_utils.SharedWeightInfo.coerce(tied_word_embedding_group())
        monkeypatch.setattr(
            mobius_utils,
            "inspect_components",
            lambda *args, **kwargs: [
                mobius_utils.ComponentInfo(
                    name="decoder",
                    role="decoder",
                    source_paths=["model.layers", "model.norm", "lm_head"],
                    shared_weights=[shared_weight],
                ),
                mobius_utils.ComponentInfo(
                    name="embedding",
                    role="embedding",
                    source_paths=["model.embed_tokens"],
                    shared_weights=[shared_weight],
                ),
            ],
        )

    def test_builds_tag_all_selected_hf_components(self, monkeypatch):
        monkeypatch.setattr(
            mobius_utils,
            "inspect_components",
            lambda *args, **kwargs: [
                mobius_utils.ComponentInfo(
                    name="decoder",
                    role="decoder",
                    source_paths=["model.layers", "lm_head"],
                ),
                mobius_utils.ComponentInfo(
                    name="embedding",
                    role="embedding",
                    source_paths=["model.embed_tokens"],
                ),
            ],
        )
        config = deepcopy(self.template)
        config["input_model"] = {
            "type": "HfModel",
            "config": {"model_path": "local/model"},
        }
        config["builds"] = {
            "decoder": {
                "components": ["decoder"],
                "pipeline": ["convert"],
            },
            "embedding": {
                "components": ["embedding"],
                "pipeline": ["convert"],
            },
        }

        expanded = expand_builds(config)

        for build in expanded.values():
            attributes = build["input_model"]["config"]["model_attributes"]
            assert attributes["workflow_components"] == ["decoder", "embedding"]

    @pytest.mark.parametrize(
        ("embedding_options", "decoder_options", "mixed_default", "planned", "deferred", "error"),
        [
            ({}, {}, None, ["word_embeddings"], ["word_embeddings"], None),
            ({"embeds": False}, {}, None, [], [], None),
            ({"modules_to_not_convert": ["model.embed_tokens"]}, {}, None, [], [], None),
            ({}, {}, {"embeds": False}, [], [], None),
            ({}, {"lm_head": False}, None, ["word_embeddings"], [], None),
            (
                {"bits": 8},
                {"overrides": {"lm_head": {"bits": 8}}},
                None,
                ["word_embeddings"],
                ["word_embeddings"],
                None,
            ),
            ({"bits": 8}, {}, None, None, None, "incompatible quantization layouts"),
            (
                {"overrides": {"model.embed_tokens": {"bits": 8}}},
                {},
                None,
                None,
                None,
                "incompatible quantization layouts",
            ),
        ],
    )
    def test_builds_plan_only_quantized_shared_owners(
        self, monkeypatch, embedding_options, decoder_options, mixed_default, planned, deferred, error
    ):
        self._patch_tied_components(monkeypatch)
        config = deepcopy(self.template)
        config["input_model"] = {
            "type": "HfModel",
            "config": {
                "model_path": "local/model",
                **(
                    {"model_attributes": {"mixed_precision_info": {"default": mixed_default, "overrides": {}}}}
                    if mixed_default is not None
                    else {}
                ),
            },
        }
        config["passes"] = {
            "decoder_quant": {"type": "KQuant", **decoder_options},
            "embedding_quant": {"type": "KQuant", **embedding_options},
        }
        config["builds"] = {
            "decoder": {
                "components": ["decoder"],
                "pipeline": ["decoder_quant"],
            },
            "embedding": {
                "components": ["embedding"],
                "pipeline": ["embedding_quant"],
            },
        }

        if error is not None:
            with pytest.raises(ValueError, match=error):
                expand_builds(config)
            return

        expanded = expand_builds(config)

        for build in expanded.values():
            attributes = build["input_model"]["config"]["model_attributes"]
            assert attributes["workflow_components"] == ["decoder", "embedding"]
            assert attributes["workflow_planned_shared_weights"] == planned
            assert attributes["workflow_planned_deferred_shared_weights"] == deferred

    @pytest.mark.parametrize("mp_disables_embeds", [False, True])
    def test_builds_assemble_auto_selected_tied_weights(self, monkeypatch, tmp_path, mp_disables_embeds):
        from olive.common.quant.tensor import QuantTensor
        from olive.model import HfModelHandler
        from test.passes.pytorch.test_quantization_utils import make_local_tiny_dense_llama

        source = tmp_path / "source"
        make_local_tiny_dense_llama(source, tie_word_embeddings=True)
        self._patch_tied_components(monkeypatch)
        output_dir = tmp_path / "assembled"
        model_attributes = (
            {"mixed_precision_info": {"default": {"embeds": False}, "overrides": {}}} if mp_disables_embeds else {}
        )
        config = {
            "input_model": {
                "type": "HfModel",
                "config": {"model_path": str(source), "model_attributes": model_attributes},
            },
            "passes": {
                "decoder_kquant": {
                    "type": "KQuant",
                    "bits": 4,
                    "group_size": 16,
                    "sym": True,
                    "overrides": {"lm_head": {"bits": 8}},
                },
                "embedding_kquant": {
                    "type": "KQuant",
                    "bits": 8,
                    "group_size": 16,
                    "sym": True,
                },
            },
            "builds": {
                "decoder": {
                    "components": ["decoder"],
                    "pipeline": ["decoder_kquant"],
                },
                "embedding": {
                    "components": ["embedding"],
                    "pipeline": ["embedding_kquant"],
                },
            },
            "engine": {
                "output_dir": str(output_dir),
                "cache_dir": str(tmp_path / "cache"),
                "evaluate_input_model": False,
            },
            "max_concurrent_builds": 1,
        }

        olive_run(config)

        checkpoint = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert ("model.embed_tokens.weight_qweight" in checkpoint["weight_map"]) is not mp_disables_embeds
        assert ("lm_head.weight_qweight" in checkpoint["weight_map"]) is mp_disables_embeds
        loaded = HfModelHandler(model_path=str(output_dir)).load_model()
        assert loaded.config.quantization_config.lm_head is True
        assert loaded.config.quantization_config.embeds is not mp_disables_embeds
        assert loaded.config.quantization_config.tie_word_embeddings is not mp_disables_embeds
        embedding = loaded.get_input_embeddings().weight
        assert (embedding is loaded.get_output_embeddings().weight) is not mp_disables_embeds
        assert isinstance(embedding.data, QuantTensor) is not mp_disables_embeds

    def test_builds_reject_skipped_assembly_with_deferred_shared_weights(self, monkeypatch, tmp_path):
        self._patch_tied_components(monkeypatch)
        config = deepcopy(self.template)
        config["input_model"] = {"type": "HfModel", "config": {"model_path": "local/model"}}
        config["engine"]["output_dir"] = str(tmp_path / "assembled")
        config["passes"] = {
            "decoder_quant": {"type": "KQuant"},
            "embedding_quant": {"type": "KQuant"},
            "convert": {"type": "OnnxConversion"},
        }
        config["builds"] = {
            "decoder": {"components": ["decoder"], "pipeline": ["decoder_quant"]},
            "embedding": {"components": ["embedding"], "pipeline": ["embedding_quant"]},
            "onnx": {"pipeline": ["convert"]},
        }
        output = MagicMock()
        output.has_output_model.return_value = True

        with (
            patch.object(sys.modules[olive_run.__module__], "_run_single", return_value=output),
            patch("olive.workflows.run.hf_component_assembly.try_assemble_hf_component_builds", return_value=None),
            pytest.raises(RuntimeError, match="Deferred shared weights require automatic"),
        ):
            olive_run(config)

    def test_builds_components_on_non_composite_input_raises(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "broken": {
                "pipeline": ["convert"],
                "output_dir": "out/broken",
                "components": ["text_encoder"],
            },
        }
        with pytest.raises(ValueError, match="select_components is only supported"):
            olive_run(config)

    def test_builds_fail_when_engine_produces_no_output_model(self):
        run_mock, _, engine_run_patch, acc_patch = self._patch_engine_and_acc()
        run_mock.return_value.has_output_model.return_value = False
        config = deepcopy(self.template)
        config["builds"] = {
            "empty": {
                "pipeline": ["convert"],
                "output_dir": "out/empty",
            }
        }

        with engine_run_patch, acc_patch, pytest.raises(RuntimeError, match="produced no output model"):
            olive_run(config)

    def test_builds_components_unknown_name_raises(self):
        composite_input = {
            "type": "CompositeModel",
            "config": {
                "model_components": [
                    {"type": "ONNXModel", "config": {"model_path": "a.onnx"}},
                    {"type": "ONNXModel", "config": {"model_path": "b.onnx"}},
                ],
                "model_component_names": ["text_encoder", "unet"],
            },
        }
        config = deepcopy(self.template)
        config["input_model"] = composite_input
        config["builds"] = {
            "bad": {
                "pipeline": ["convert"],
                "output_dir": "out/bad",
                "components": ["no_such_component"],
            },
        }
        with pytest.raises(ValueError, match="Unknown component"):
            olive_run(config)

    def test_builds_directory_composite_input_runs_per_component(self, tmp_path):
        # An export directory loads as a CompositeModel; subfolder names become
        # component names, and sibling builds optimize each component.
        for name in ["decoder", "vision_encoder"]:
            comp_dir = tmp_path / "exported_pkg" / name
            comp_dir.mkdir(parents=True)
            (comp_dir / "model.onnx").write_bytes(b"onnx")

        run_mock, _, engine_run_patch, acc_patch = self._patch_engine_and_acc()
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(tmp_path / "exported_pkg")}}
        config["engine"]["output_dir"] = str(tmp_path / "assembled")
        config["builds"] = {
            "decoder": {"components": ["decoder"], "pipeline": ["convert"], "output_dir": "out/decoder"},
            "vision_encoder": {"components": ["vision_encoder"], "pipeline": ["convert"], "output_dir": "out/vision"},
        }
        with (
            engine_run_patch,
            acc_patch,
            patch("olive.workflows.run.component_assembly.try_assemble_component_builds") as assemble_mock,
        ):
            result = olive_run(config)
        assert set(result) == {"decoder", "vision_encoder"}
        assert run_mock.call_count == 2
        assemble_mock.assert_called_once()

    def test_builds_create_suffixed_output_paths_as_directories(self, tmp_path):
        config = deepcopy(self.template)
        config["max_concurrent_builds"] = 2
        config["builds"] = {
            "first": {
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "first.onnx"),
            },
            "second": {
                "pipeline": ["tune"],
                "output_dir": str(tmp_path / "second.onnx"),
            },
        }

        def run_build(_, run_config):
            output_dir = Path(run_config.engine.output_dir)
            return output_dir, output_dir.is_dir()

        with patch.object(sys.modules[olive_run.__module__], "_run_single", side_effect=run_build):
            result = olive_run(config)

        assert result["first"][1] is True
        assert result["second"][1] is True
        assert result["first"][0].parent == result["second"][0].parent

    def test_builds_prevalidate_all_configs_before_running(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "valid": {"pipeline": ["convert"], "output_dir": "out/valid"},
            "broken": {
                "pipeline": ["convert"],
                "output_dir": "out/broken",
                "search_strategy": True,
            },
        }

        with (
            patch("olive.engine.engine.Engine.run") as run_mock,
            pytest.raises(ValueError, match="Invalid build 'broken'"),
        ):
            olive_run(config)
        run_mock.assert_not_called()

    def test_builds_union_required_packages_across_expanded_configs(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/first"},
            "second": {"pipeline": ["tune"], "output_dir": "out/second"},
        }
        run_module = sys.modules[olive_run.__module__]

        with (
            patch.object(run_module, "get_required_packages", side_effect=[{"package-a"}, {"package-b"}]),
            patch.object(run_module, "generate_files_from_packages") as generate_mock,
        ):
            olive_run(config, list_required_packages=True)

        generate_mock.assert_called_once_with({"package-a", "package-b"}, "olive_requirements.txt")

    def test_builds_reject_non_local_host(self):
        config = deepcopy(self.template)
        config["systems"] = {
            "docker_system": {
                "type": "Docker",
                "dockerfile": "Dockerfile",
                "build_context_path": ".",
            }
        }
        config["builds"] = {
            "docker": {"pipeline": ["convert"], "output_dir": "out/docker"},
        }
        config["engine"]["host"] = "docker_system"

        with (
            patch("olive.engine.engine.Engine.run") as run_mock,
            pytest.raises(ValueError, match="only LocalSystem hosts"),
        ):
            olive_run(config)
        run_mock.assert_not_called()

    def test_builds_report_build_name_when_parallel_execution_fails(self):
        config = deepcopy(self.template)
        config["max_concurrent_builds"] = 2
        config["builds"] = {
            "good": {"pipeline": ["convert"], "output_dir": "out/good"},
            "bad": {"pipeline": ["tune"], "output_dir": "out/bad"},
        }
        barrier = Barrier(2)

        def run_build(_, run_config):
            barrier.wait(timeout=2)
            if run_config.workflow_id.endswith("_bad"):
                raise ValueError("bad build")
            return run_config.workflow_id

        with (
            patch.object(sys.modules[olive_run.__module__], "_run_single", side_effect=run_build),
            pytest.raises(RuntimeError, match=r"Build\(s\) \['bad'\] failed.*bad build"),
        ):
            olive_run(config)

    @pytest.mark.parametrize(
        ("max_concurrent_builds", "pass_type", "expected_max_active"),
        [
            (None, None, 3),
            (2, None, 2),
            (2, "KQuant", 1),
            (2, "Rtn", 1),
            (2, "DoRA", 1),
            (2, "Gptq", 1),
            (2, "IncDynamicQuantization", 1),
            (2, "IncQuantization", 1),
            (2, "IncStaticQuantization", 1),
            (2, "LoftQ", 1),
            (2, "LoHa", 1),
            (2, "LoKr", 1),
            (2, "LoRA", 1),
            (2, "OnnxDiscrepancyCheck", 1),
            (2, "OpenVINOConversion", 1),
            (2, "OpenVINOOptimumConversion", 1),
            (2, "QuaRot", 1),
            (2, "EPContextBinaryGenerator", 1),
            (2, "QairtEncapsulation", 1),
            (2, "QairtGenAIBuilder", 1),
            (2, "QairtPipelinePass", 1),
            (2, "QLoRA", 1),
            (2, "QuarkQuantization", 1),
            (2, "QuarkQuantizationVitisAI", 1),
            (2, "SaveTestModelConfig", 1),
            (2, "SDLoRA", 1),
            (2, "SliceGPT", 1),
            (2, "SpinQuant", 1),
        ],
    )
    def test_builds_limit_concurrency(self, max_concurrent_builds, pass_type, expected_max_active):
        config = deepcopy(self.template)
        if max_concurrent_builds is not None:
            config["max_concurrent_builds"] = max_concurrent_builds
        if pass_type:
            config["passes"]["serial"] = {"type": pass_type}
        config["builds"] = {
            "first": {
                "pipeline": ["serial" if pass_type else "convert"],
                "output_dir": "out/first",
            },
            "second": {"pipeline": ["tune"], "output_dir": "out/second"},
            "third": {"pipeline": ["convert"], "output_dir": "out/third"},
        }

        active = 0
        max_active = 0
        lock = Lock()

        def run_build(_, run_config):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                sleep(0.05)
                return run_config.workflow_id
            finally:
                with lock:
                    active -= 1

        with patch.object(sys.modules[olive_run.__module__], "_run_single", side_effect=run_build):
            result = olive_run(config)

        assert set(result) == {"first", "second", "third"}
        assert max_active == expected_max_active
