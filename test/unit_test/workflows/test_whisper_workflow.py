# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from test.unit_test.workflows.whisper_utils import (
    decoder_dummy_inputs,
    encoder_decoder_init_dummy_inputs,
    get_dec_io_config,
    get_decoder,
    get_encdec_io_config,
    get_encoder_decoder_init,
    whisper_audio_decoder_dataloader,
)
from urllib import request

import pytest

from olive.workflows import run as olive_run


@pytest.fixture(name="audio_data")
def download_audio_test_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    test_audio_name = "1272-141231-0002.mp3"
    test_audio_url = (
        "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/" + test_audio_name
    )
    test_audio_path = data_dir / test_audio_name
    if not test_audio_path.exists():
        request.urlretrieve(test_audio_url, test_audio_path)

    return str(data_dir)


@pytest.fixture(name="whisper_config")
def prepare_whisper_config(audio_data, tmp_path):
    return {
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "hf_config": {
                    "model_class": "WhisperForConditionalGeneration",
                    "model_name": "openai/whisper-tiny.en",
                    "components": [
                        {
                            "name": "encoder_decoder_init",
                            "io_config": get_encdec_io_config,
                            "component_func": get_encoder_decoder_init,
                            "dummy_inputs_func": encoder_decoder_init_dummy_inputs,
                        },
                        {
                            "name": "decoder",
                            "io_config": get_dec_io_config,
                            "component_func": get_decoder,
                            "dummy_inputs_func": decoder_dummy_inputs,
                        },
                    ],
                }
            },
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]},
            }
        },
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {
                        "name": "latency",
                        "type": "latency",
                        "sub_types": [{"name": "avg", "priority": 1}],
                        "user_config": {"data_dir": audio_data, "dataloader_func": whisper_audio_decoder_dataloader},
                    }
                ]
            }
        },
        "passes": {
            "conversion": {"type": "OnnxConversion", "config": {"target_opset": 17}},
            "transformers_optimization": {
                "type": "OrtTransformersOptimization",
                "disable_search": True,
                "config": {"optimization_options": {"use_multi_head_attention": True}, "use_gpu": False},
            },
            "inc_dynamic_quantization": {
                "type": "IncDynamicQuantization",
                "disable_search": True,
                "config": {
                    "workspace": str(tmp_path / "workspace"),
                },
            },
            "insert_beam_search": {
                "type": "InsertBeamSearch",
                "config": {"use_forced_decoder_ids": False, "fp16": False},
            },
            "prepost": {
                "type": "AppendPrePostProcessingOps",
                "config": {
                    "tool_command": "whisper",
                    "tool_command_args": {"model_name": "openai/whisper-tiny.en", "use_audio_decoder": True},
                    "target_opset": 17,
                },
            },
        },
        "engine": {
            "log_severity_level": 0,
            "search_strategy": False,
            "host": "local_system",
            "target": "local_system",
            "evaluator": "common_evaluator",
            "evaluate_input_model": False,
            "clean_cache": True,
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "models"),
            "output_name": "whisper_cpu_fp32",
        },
    }


def check_output(footprints):
    """Check if the search output is valid."""
    assert footprints, "footprints is empty. The search must have failed for all accelerator specs."
    for footprint in footprints.values():
        assert footprint.nodes
        for v in footprint.nodes.values():
            assert all(metric_result.value > 0 for metric_result in v.metrics.value.values())


def test_whisper_run(tmp_path, whisper_config):
    from packaging import version
    from transformers import __version__ as transformers_version

    if version.parse(transformers_version) >= version.parse("4.36.0"):
        whisper_config["input_model"]["config"]["hf_config"]["from_pretrained_args"] = {"attn_implementation": "eager"}
    result = olive_run(whisper_config)
    check_output(result)
    shutil.rmtree(tmp_path, ignore_errors=True)
