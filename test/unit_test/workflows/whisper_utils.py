from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, WhisperConfig, file_utils

if TYPE_CHECKING:
    from olive.model.handler.pytorch import PyTorchModelHandler


def get_decoder(olive_model: "PyTorchModelHandler"):
    # model is WhisperForConditionalGeneration
    model = olive_model.load_model()
    return WhisperDecoder(model, model.config)


def get_dec_io_config(olive_model: "PyTorchModelHandler"):
    # Fix past disappearing bug - duplicate first past entry
    # input_list.insert(2, input_list[2])
    config = olive_model.get_hf_model_config()
    past_names = PastKeyValuesHelper.get_past_names(config.decoder_layers, present=False)
    present_names = PastKeyValuesHelper.get_past_names(config.decoder_layers, present=True)
    present_self_names = present_names[: 2 * config.decoder_layers]

    input_past_names = past_names
    output_present_names = present_self_names
    output_names = ["logits", *output_present_names]

    input_names = ["input_ids"]
    input_names.extend(input_past_names)

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    for name in input_past_names:
        dynamic_axes[name] = {
            0: "batch_size",
            2: "past_decode_sequence_length" if "self" in name else "encode_sequence_length",
        }

    for name in output_present_names:
        if "cross" in name:
            dynamic_axes[name] = {0: "batch_size", 2: "encode_sequence_length"}
        else:  # self attention past state
            dynamic_axes[name] = {
                0: "batch_size",
                2: "past_decode_sequence_length + 1",
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


def decoder_dummy_inputs(olive_model: "PyTorchModelHandler"):
    inputs = WhisperDecoderInputs.create_dummy(
        olive_model.get_hf_model_config(),
        batch_size=2,
        encode_sequence_length=3000,
        past_decode_sequence_length=5,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


def get_encdec_io_config(olive_model: "PyTorchModelHandler"):
    # model is WhisperEncoderDecoderInit
    model = olive_model.load_model()
    use_decoder_input_ids = True

    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=use_decoder_input_ids,
        device="cpu",
        use_int32_inputs=True,
    )

    out = model(inputs.encoder_input_ids, inputs.decoder_input_ids)
    present = out[2]
    present_names = PastKeyValuesHelper.get_input_names(present, encoder=True)

    output_names = ["logits", "encoder_hidden_states", *present_names]

    input_names = ["encoder_input_ids"]

    # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference.
    # We use a workaround here: first use dim_param str(model.config.encoder_attention_heads) for num_heads,
    # and later change to dim_value.
    num_heads = str(model.config.encoder_attention_heads)
    hidden_size = str(model.config.d_model)
    head_size = str(model.config.d_model // model.config.encoder_attention_heads)
    dynamic_axes = {
        "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_hidden_states": {
            0: "batch_size",
            1: "encode_sequence_length",
            2: hidden_size,
        },
        "logits": {
            0: "batch_size",
            1: "decode_sequence_length",
        },
    }

    if use_decoder_input_ids:
        input_names.append("decoder_input_ids")
        dynamic_axes["decoder_input_ids"] = {
            0: "batch_size",
            1: "decode_sequence_length",
        }

    for name in present_names:
        if "cross" in name:
            dynamic_axes[name] = {
                0: "batch_size",
                1: num_heads,
                2: "encode_sequence_length",
                3: head_size,
            }

        else:  # self attention past state
            dynamic_axes[name] = {
                0: "batch_size",
                1: num_heads,
                2: "decode_sequence_length",
                3: head_size,
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
        "string_to_int_dim_params": [num_heads, hidden_size, head_size],
    }


def get_encoder_decoder_init(olive_model: "PyTorchModelHandler"):
    # model is WhisperForConditionalGeneration
    model = olive_model.load_model()
    return WhisperEncoderDecoderInit(
        model,
        model,
        model.config,
        decoder_start_token_id=None,
    )


def encoder_decoder_init_dummy_inputs(olive_model: "PyTorchModelHandler"):
    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        olive_model.get_hf_model_config(),
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=True,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


def whisper_audio_decoder_dataloader(data_dir, batch_size, *args, **kwargs):
    return WhisperDataset(data_dir=data_dir, use_audio_decoder=True)


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
                past.append(torch.rand(self_attention_past_shape, dtype=float_type, device=device))  # noqa: PERF401

            for _ in range(2 * num_layers):
                past.append(torch.rand(cross_attention_past_shape, dtype=float_type, device=device))  # noqa: PERF401
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


class WhisperEncoder(torch.nn.Module):
    """Whisper encoder outputs only the last hidden state."""

    def __init__(self, encoder, config: WhisperConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_features):
        return self.encoder.model.encoder(input_features)[0]


class WhisperEncoderInputs:
    def __init__(self, input_features):
        self.input_ids: torch.FloatTensor = input_features
        # HF Whisper model doesn't support Attention Mask functionality

    @staticmethod
    def create_dummy(
        batch_size: int, sequence_length: int, feature_size: int, device: torch.device, use_int32_inputs: bool
    ):
        """Create dummy inputs for Whisper encoder.

        Args:
            batch_size (int): batch size
            sequence_length (int): sequence length
            feature_size (int): feature size for spectrogram input
            device (torch.device): device of output tensors
            use_int32_inputs (bool): whether to use int32 inputs

        Returns:
            WhisperEncoderInputs: dummy inputs for encoder

        """
        input_features = torch.randn(
            size=(batch_size, feature_size, sequence_length),
            device=device,
        )
        return WhisperEncoderInputs(input_features)

    def to_list(self) -> List:
        if self.input_features is None:
            return []
        return [self.input_features]


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


class PastKeyValuesHelper:
    """Helper functions to process past key values for encoder-decoder model."""

    @staticmethod
    def get_past_names(num_layers, present: bool = False):
        past_self_names = []
        past_cross_names = []
        for i in range(num_layers):
            past_self_names.extend(
                [f"present_key_self_{i}", f"present_value_self_{i}"]
                if present
                else [f"past_key_self_{i}", f"past_value_self_{i}"]
            )
            past_cross_names.extend(
                [f"present_key_cross_{i}", f"present_value_cross_{i}"]
                if present
                else [f"past_key_cross_{i}", f"past_value_cross_{i}"]
            )
        return past_self_names + past_cross_names

    @staticmethod
    def group_by_self_or_cross(present_key_values):
        """Split present state from grouped by layer to grouped by self/cross attention.

        Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
                (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
        After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...),
            (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

        """
        present_self = []
        present_cross = []
        for _i, present_layer_i in enumerate(present_key_values):
            assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
            (
                present_key_self,
                present_value_self,
                present_key_cross,
                present_value_cross,
            ) = present_layer_i
            present_self.extend([present_key_self, present_value_self])
            present_cross.extend([present_key_cross, present_value_cross])
        return present_self, present_cross

    @staticmethod
    def group_by_layer(past, num_layers):
        """Reorder past state from grouped by self/cross attention to grouped by layer.

        Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...,
                past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
        After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
               (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),
        """
        assert len(past) == 4 * num_layers
        return tuple(
            [
                past[2 * i],
                past[2 * i + 1],
                past[2 * num_layers + 2 * i],
                past[2 * num_layers + 2 * i + 1],
            ]
            for i in range(num_layers)
        )

    @staticmethod
    def back_group_by_layer(past_key_values: Tuple[Tuple[torch.Tensor]]):
        """Categorize past_key_values from self and cross attention to layer by layer.

        Reorder past state from grouped by self/cross attention to grouped by layer.
        Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...,
                past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
        After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
                (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),

        Args:
            past_key_values: From past_key_values of a model (group by self and cross attention)

        Returns:
            past_tuples: present key and values grouped by layer.

        """
        past_tuples = ()
        half_idx = len(past_key_values) // 2
        for i in range(len(past_key_values) // 4):
            idx = 2 * i
            past_tuples += (
                (
                    past_key_values[idx],
                    past_key_values[idx + 1],
                    past_key_values[half_idx + idx],
                    past_key_values[half_idx + idx + 1],
                ),
            )
        return past_tuples

    @staticmethod
    def group_by_self_and_cross(present_key_values: Tuple[torch.Tensor], concat: bool = False):
        """Categorize present_key_values into self and cross attention.

        Split present state from grouped by layer to grouped by self/cross attention.
        Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
                (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
        After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...),
                (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

        Args:
            present_key_values: From past_key_values of a model (group by layer)
            concat: If concat self attention with cross attention key/value to return

        Returns:
            present_self (Tuple[torch.Tensor]): present key and values from self attention
            present_cross (Tuple[torch.Tensor]): present key and values from cross attention

        """
        present_self: List[torch.Tensor] = []
        present_cross: List[torch.Tensor] = []
        for _, present_layer_i in enumerate(present_key_values):
            assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
            present_key_self, present_value_self, present_key_cross, present_value_cross = present_layer_i
            present_self.extend([present_key_self, present_value_self])
            present_cross.extend([present_key_cross, present_value_cross])
        if concat:
            return present_self + present_cross
        else:
            return present_self, present_cross

    @staticmethod
    def get_input_names(past_key_values: Tuple[Tuple[torch.Tensor]], encoder=True):
        """Process input names of model wrapper.

        Args:
            past_key_values: Consider `self` and `cross` past_key_values
            encoder: If encoder or decoder

        Returns:
            names (List[string]): input names

        """
        names = []
        num_layers = len(past_key_values) // 4 if encoder else len(past_key_values)
        prefix = "past_" if not encoder else "present_"
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}"]])
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_cross_{i}", f"value_cross_{i}"]])
        return names


class WhisperDataset:
    SAMPLE_RATE = 16000
    N_FFT = 400
    N_MELS = 80
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = N_SAMPLES // HOP_LENGTH

    def __init__(
        self,
        data_dir: str,
        use_audio_decoder: bool = True,
        file_ext: str = ".mp3",
        language: str = "english",
        task: str = "transcribe",
    ):
        data_dir = Path(data_dir)
        audio_files = list(data_dir.glob(f"*{file_ext}"))
        audio_files.sort(key=lambda x: x.name)
        assert len(audio_files) > 0, f"No audio files found in {data_dir}"

        self.data = []
        for audio_file in audio_files:
            if use_audio_decoder:
                with open(audio_file, "rb") as _f:
                    audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
                audio_input_name = "audio_stream"
            else:
                import librosa

                audio_blob, _ = librosa.load(audio_file)
                audio_input_name = "audio_pcm"

            audio_blob = np.expand_dims(audio_blob, axis=0)  # add a batch_size
            inputs = {
                audio_input_name: audio_blob,
                "max_length": np.asarray([200], dtype=np.int32),
                "min_length": np.asarray([0], dtype=np.int32),
                "num_beams": np.asarray([2], dtype=np.int32),
                "num_return_sequences": np.asarray([1], dtype=np.int32),
                "length_penalty": np.asarray([1.0], dtype=np.float32),
                "repetition_penalty": np.asarray([1.0], dtype=np.float32),
                # attention_mask only used when version < 1.16.0
                "attention_mask": np.zeros((1, self.N_MELS, self.N_FRAMES)).astype(np.int32),
            }
            # decoder_input_ids only used when version >= 1.16.0 and multilingual is True
            model_name = "openai/whisper-tiny"
            config = AutoConfig.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
            forced_decoder_ids = [config.decoder_start_token_id, *[token[1] for token in forced_decoder_ids]]
            inputs["decoder_input_ids"] = np.asarray([forced_decoder_ids], dtype=np.int32)

            self.data.append(inputs)

        self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1  # pylint: disable=unsubscriptable-object
        return data, label

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
