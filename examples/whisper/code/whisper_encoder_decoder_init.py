# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List, Optional

import torch
from past_helper import PastKeyValuesHelper
from transformers import WhisperConfig
from whisper_decoder import WhisperDecoderInit
from whisper_encoder import WhisperEncoder, WhisperEncoderInputs


class WhisperEncoderDecoderInit(torch.nn.Module):
    """A combination of WhisperEncoder and WhisperDecoderInit."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        config: WhisperConfig,
        decoder_start_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.whisper_encoder = WhisperEncoder(encoder, config)
        self.whisper_decoder_init = WhisperDecoderInit(decoder, config, decoder_start_token_id)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
    ):
        encoder_hidden_states: torch.FloatTensor = self.whisper_encoder(encoder_input_ids)
        # Decoder out: (logits, past_key_values, encoder_hidden_state)
        decinit_out = self.whisper_decoder_init(decoder_input_ids, encoder_hidden_states)
        present_self, present_cross = PastKeyValuesHelper.group_by_self_and_cross(decinit_out[1])
        present = present_self + present_cross
        return decinit_out[0], encoder_hidden_states, present


class WhisperEncoderDecoderInitInputs:
    def __init__(self, encoder_input_ids, decoder_input_ids=None):
        self.encoder_input_ids: torch.LongTensor = encoder_input_ids
        self.decoder_input_ids: torch.LongTensor = decoder_input_ids

    @staticmethod
    def create_dummy(
        config: WhisperConfig,
        batch_size: int,
        encode_sequence_length: int,
        use_decoder_input_ids: int,
        device: torch.device,
        use_int32_inputs: bool = False,
    ):  # -> WhisperEncoderDecoderInitInputs:
        encoder_inputs: WhisperEncoderInputs = WhisperEncoderInputs.create_dummy(
            batch_size,
            sequence_length=3000,
            feature_size=config.num_mel_bins,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )
        decoder_input_ids = None
        if use_decoder_input_ids:
            dtype = torch.int32 if use_int32_inputs else torch.int64
            decoder_input_ids = torch.ones((batch_size, 2), dtype=dtype, device=device) * config.decoder_start_token_id

        return WhisperEncoderDecoderInitInputs(encoder_inputs.input_ids, decoder_input_ids)

    def to_list(self) -> List:
        input_list = [self.encoder_input_ids]
        if self.decoder_input_ids is not None:
            input_list.append(self.decoder_input_ids)
        return input_list
