from typing import Dict, Optional

import torch
from onnxruntime.transformers.models.t5.past_helper import PastKeyValuesHelper
from transformers import WhisperConfig, WhisperForConditionalGeneration, file_utils


# this is a temporary file
# will be replaced with ort whisper code when available
class WhisperEncoder(torch.nn.Module):
    """Whisper encoder outputs only the last hidden state"""

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_features, attention_mask):
        return self.encoder.model.encoder(input_features)[0]


class WhisperDecoderInit(torch.nn.Module):
    """A Whisper decoder with LM head to create initial past key values.
    This model is only called once during starting decoding.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        lm_head: torch.nn.Module,
        config: WhisperConfig,
        decoder_start_token_id: int = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config
        self.decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
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
            head_mask=encoder_attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        logits = self.decoder.proj_out(out[0])
        return logits, out.past_key_values, out.encoder_last_hidden_state


class WhisperDecoder(torch.nn.Module):
    """A Whisper decoder with LM head and past key values"""

    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    def forward(self, decoder_input_ids, encoder_attention_mask, *past):
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
            # decoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = decoder_out[0]
        present_self, _ = PastKeyValuesHelper.group_by_self_and_cross(decoder_out.past_key_values)
        return logits, present_self


class WhisperEncoderDecoderInit(torch.nn.Module):
    """A combination of WhisperEncoder and WhisperDecoderInit."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        lm_head: torch.nn.Module,
        config,
        decoder_start_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.whisper_encoder = WhisperEncoder(encoder, config)
        self.whisper_decoder_init = WhisperDecoderInit(decoder, lm_head, config, decoder_start_token_id)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
    ):
        encoder_hidden_states: torch.FloatTensor = self.whisper_encoder(encoder_input_ids, None)
        # Decoder out: (logits, past_key_values, encoder_hidden_state)
        decinit_out = self.whisper_decoder_init(decoder_input_ids, encoder_attention_mask, encoder_hidden_states)
        present_self, present_cross = PastKeyValuesHelper.group_by_self_and_cross(decinit_out[1])
        present = present_self + present_cross
        return decinit_out[0], encoder_hidden_states, present


def load_model(model_name_or_path: str) -> Dict[str, torch.nn.Module]:
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    decoder = WhisperDecoder(model, None, model.config)
    encoder_decoder_init = WhisperEncoderDecoderInit(model, model, None, model.config, decoder_start_token_id=None)
    return {"encoder_decoder_init": encoder_decoder_init, "decoder": decoder}
