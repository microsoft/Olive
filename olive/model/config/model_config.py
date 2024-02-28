# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator
from olive.model.config.registry import get_model_handler, is_valid_model_type
from olive.resource_path import create_resource_path


class ModelConfig(ConfigBase):
    """Input model config which will be used to create the model handler.

    For example, the config looks like for llama2:
    .. code-block:: json

        {
            "input_model": {
                "type": "CompositePyTorchModel",
                "config": {

                    "model_path": "llama_v2",
                    "model_components": [

                        {
                            "name": "decoder_model",
                            "type": "PyTorchModel",
                            "config": {

                                "model_script": "user_script.py",
                                "io_config": {

                                    "input_names": ["tokens", "position_ids", "attn_mask", ...],
                                    "output_names": ["logits", "attn_mask_out", ...],
                                    "dynamic_axes": {

                                        "tokens": { "0": "batch_size", "1": "seq_len" },
                                        "position_ids": { "0": "batch_size", "1": "seq_len" },
                                        "attn_mask": { "0": "batch_size", "1": "max_seq_len" },
                                        ...

                                    }

                                },
                                "model_loader": "load_decoder_model",
                                "dummy_inputs_func": "decoder_inputs"

                            }

                        },
                        {

                            "name": "decoder_with_past_model",
                            "type": "PyTorchModel",
                            "config": {

                                "model_script": "user_script.py",
                                "io_config": {

                                    "input_names": ["tokens_increment", "position_ids_increment", "attn_mask", ...],
                                    "output_names": ["logits", "attn_mask_out", ...],
                                    "dynamic_axes": {

                                        "tokens_increment": { "0": "batch_size", "1": "seq_len_increment" },
                                        "position_ids_increment": { "0": "batch_size", "1": "seq_len_increment" },
                                        "attn_mask": { "0": "batch_size", "1": "max_seq_len" },
                                        ...

                                    }

                                },
                                "model_loader": "load_decoder_with_past_model",
                                "dummy_inputs_func": "decoder_with_past_inputs"

                            }

                        }

                    ]

                }

            }

        }

    """

    type: str
    config: dict

    @validator("type")
    def validate_type(cls, v):
        if not is_valid_model_type(v):
            raise ValueError(f"Unknown model type {v}")
        return v

    def get_resource_strings(self):
        cls = get_model_handler(self.type)
        resource_keys = cls.get_resource_keys()
        return {k: v for k, v in self.config.items() if k in resource_keys}

    def get_resource_paths(self):
        resources = self.get_resource_strings()
        return {k: create_resource_path(v) for k, v in resources.items()}

    def create_model(self):
        cls = get_model_handler(self.type)
        return cls(**self.config)
