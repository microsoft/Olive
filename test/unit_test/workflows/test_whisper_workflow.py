from pathlib import Path
from test.unit_test.workflows.whisper_utils import (
    decoder_dummy_inputs,
    encoder_decoder_init_dummy_inputs,
    get_dec_io_config,
    get_decoder,
    get_encdec_io_config,
    get_encoder_decoder_init,
    whisper_audio_decoder_dataloader,
)

import pytest

from olive.workflows import run as olive_run


@pytest.fixture(name="whisper_config")
def prepare_whisper_config():
    data_dir = Path(__file__).parents[3] / "examples" / "whisper" / "data"
    data_dir = str(data_dir)
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
        "systems": {"local_system": {"type": "LocalSystem", "config": {"accelerators": ["cpu"]}}},
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {
                        "name": "latency",
                        "type": "latency",
                        "sub_types": [{"name": "avg", "priority": 1}],
                        "user_config": {"data_dir": data_dir, "dataloader_func": whisper_audio_decoder_dataloader},
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
            "execution_providers": ["CPUExecutionProvider"],
            "clean_cache": True,
            "cache_dir": "cache",
            "output_dir": "models",
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


def test_whisper_run(whisper_config):
    result = olive_run(whisper_config)
    check_output(result)
