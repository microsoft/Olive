import torch
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection


def text_encoder_inputs(batch_size, torch_dtype):
    return {
        "input_ids": torch.zeros((batch_size, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def _dummy_inputs(model=None):
    return text_encoder_inputs(1, torch.int64)


def _model_loader(model_name):
    return CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2")


def _io_config(model):
    return {
        "input_names": ["input_ids", "output_hidden_states"],
        "output_names": [
            "text_embeds",
            "last_hidden_state",
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
            "hidden_states.13",
            "hidden_states.14",
            "hidden_states.15",
            "hidden_states.16",
            "hidden_states.17",
            "hidden_states.18",
            "hidden_states.19",
            "hidden_states.20",
            "hidden_states.21",
            "hidden_states.22",
            "hidden_states.23",
            "hidden_states.24",
            "hidden_states.25",
            "hidden_states.26",
            "hidden_states.27",
            "hidden_states.28",
            "hidden_states.29",
            "hidden_states.30",
            "hidden_states.31",
            "hidden_states.32",
        ],
        "dynamic_axes": {
            "input_ids": {"0": "batch_size", "1": "sequence_length"},
            "text_embeds": {"0": "batch_size"},
            "last_hidden_state": {"0": "batch_size", "1": "sequence_length"},
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
            "hidden_states.13": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.14": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.15": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.16": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.17": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.18": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.19": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.20": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.21": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.22": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.23": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.24": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.25": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.26": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.27": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.28": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.29": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.30": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.31": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states.32": {"0": "batch_size", "1": "sequence_length"},
        },
    }
