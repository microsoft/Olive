import torch
from transformers.models.clip.modeling_clip import CLIPTextModel


def text_encoder_inputs(batch_size, torch_dtype):
    return {
        "input_ids": torch.zeros((batch_size, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def _dummy_inputs(model=None):
    return text_encoder_inputs(1, torch.int32)


def _model_loader(model_name):
    return CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")


def _io_config(model):
    return {
        "input_names": ["input_ids", "output_hidden_states"],
        "output_names": [
            "last_hidden_state",
            "pooler_output",
            "hidden_states.0",
            "hidden_states.1",
            "hidden_states.2",
            "hidden_states.3",
            "hidden_states.4",
            "hidden_states.5",
            "hidden_states.6",
            "hidden_states.7",
            "hidden_states.8",
            "hidden_states.9",
            "hidden_states.10",
            "hidden_states.11",
            "hidden_states.12",
        ],
        "dynamic_axes": {
            "input_ids": {"0": "batch_size", "1": "sequence_length"},
            "last_hidden_state": {"0": "batch_size", "1": "sequence_length"},
            "pooler_output": {"0": "batch_size"},
            "hidden_states.0": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.1": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.2": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.3": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.4": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.5": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.6": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.7": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.8": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.9": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.10": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.11": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.12": {"0": "batch_size", "1": "sequence_length"},
        },
    }
