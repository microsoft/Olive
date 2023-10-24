# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List, Union

import torch
from past_helper import PastKeyValuesHelper
from transformers import WhisperConfig, file_utils


class WhisperDecoderInit(torch.nn.Module):
    """A Whisper decoder to create initial past key values.

    This model is only called once during starting decoding.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        config: WhisperConfig,
        decoder_start_token_id: int = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor,
    ):
        encoder_outputs = file_utils.ModelOutput()
        encoder_outputs["last_hidden_state"] = encoder_hidden_states
        encoder_outputs["hidden_states"] = None
        encoder_outputs["attentions"] = None

        out = self.decoder.model(
            None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        logits = self.decoder.proj_out(out[0])
        return logits, out.past_key_values, out.encoder_last_hidden_state


class WhisperDecoder(torch.nn.Module):
    """A Whisper decoder and past key values."""

    def __init__(self, decoder, config):
        super().__init__()
        self.decoder = decoder
        self.config = config

    def forward(self, decoder_input_ids, *past):
        encoder_outputs = file_utils.ModelOutput()
        dummy_encoder_hidden_states = torch.randn((decoder_input_ids.shape[0], 3000, int(self.config.d_model)))
        encoder_outputs["last_hidden_state"] = dummy_encoder_hidden_states
        encoder_outputs["hidden_states"] = dummy_encoder_hidden_states
        encoder_outputs["attentions"] = None
        if len(past) == 0:
            past_key_values = None
        else:
            past_key_values = PastKeyValuesHelper.back_group_by_layer(past)

        decoder_out = self.decoder(
            None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = decoder_out[0]
        present_self, _ = PastKeyValuesHelper.group_by_self_and_cross(decoder_out.past_key_values)
        return logits, present_self


class WhisperDecoderInputs:
    def __init__(
        self,
        decoder_input_ids,
        past_key_values=None,
    ):
        self.decoder_input_ids: torch.LongTensor = decoder_input_ids
        self.past_key_values: Union[List[torch.FloatTensor], List[torch.HalfTensor], None] = past_key_values

    @staticmethod
    def create_dummy(
        config: WhisperConfig,
        batch_size: int,
        encode_sequence_length: int,
        past_decode_sequence_length: int,
        device: torch.device,
        float16: bool = False,
        use_int32_inputs: bool = False,
    ):  # -> WhisperDecoderInputs:
        """Create dummy inputs for WhisperDecoder.

        Args:
            config: config
            batch_size (int): batch size
            encode_sequence_length (int): sequence length of input_ids for encoder
            past_decode_sequence_length (int): past sequence length of input_ids for decoder
            device (torch.device): device of output tensors
            float16 (bool): whether the model uses float32 or float16 in input
            use_int32_inputs(bool): whether use int32 instead of int64 for some inputs

        Returns:
            WhisperDecoderInputs: dummy inputs for decoder
        """
        num_attention_heads: int = config.encoder_attention_heads
        num_layers: int = config.decoder_layers  # + config.encoder_layers
        vocab_size: int = config.vocab_size

        # Use head_size, use hidden_size / num_attention_heads here.
        # For example, whisper-large, d_model=1280 and num_heads=20
        head_size: int = config.d_model // config.encoder_attention_heads

        sequence_length: int = 1  # fixed for decoding
        decoder_input_ids = torch.randint(
            low=0,
            high=vocab_size - 1,
            size=(batch_size, sequence_length),
            dtype=(torch.int32 if use_int32_inputs else torch.int64),
            device=device,
        )

        float_type = torch.float16 if float16 else torch.float32

        if past_decode_sequence_length > 0:
            self_attention_past_shape = [
                batch_size,
                num_attention_heads,
                past_decode_sequence_length,
                head_size,
            ]
            cross_attention_past_shape = [
                batch_size,
                num_attention_heads,
                encode_sequence_length,
                head_size,
            ]

            past = []
            for _ in range(2 * num_layers):
                past.append(torch.rand(self_attention_past_shape, dtype=float_type, device=device))

            for _ in range(2 * num_layers):
                past.append(torch.rand(cross_attention_past_shape, dtype=float_type, device=device))
        else:
            past = None

        return WhisperDecoderInputs(decoder_input_ids, past)

    def to_list(self) -> List:
        input_list = [self.decoder_input_ids]
        if self.past_key_values:
            input_list.extend(self.past_key_values)
        return input_list

    def to_fp32(self):
        past = [p.to(dtype=torch.float32) for p in self.past_key_values] if self.past_key_values else None
        return WhisperDecoderInputs(
            self.decoder_input_ids.clone(),
            past,
        )
