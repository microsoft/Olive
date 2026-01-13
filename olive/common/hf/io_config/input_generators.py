# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from olive.common.hf.io_config.normalized_config import (
    NormalizedConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)
from olive.common.hf.io_config.tasks import TaskType

DEFAULT_DUMMY_SHAPES = {
    "batch_size": 2,
    "sequence_length": 16,
    "num_choices": 4,
    # image
    "width": 64,
    "height": 64,
    "num_channels": 3,
    "point_batch_size": 3,
    "nb_points_per_image": 2,
    "visual_seq_length": 16,
    # audio
    "feature_size": 80,
    "nb_max_frames": 3000,
    "audio_sequence_length": 16000,
}


class DtypeMapper:
    @classmethod
    def np(cls, dtype):
        import numpy as np

        mapping = {
            "fp32": np.float32,
            "fp16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int8": np.int8,
            "bool": bool,
        }
        return mapping[dtype]

    @classmethod
    def pt(cls, dtype):
        import torch

        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        return mapping[dtype]


class DummyInputGenerator(ABC):
    """Generate dummy inputs for the supported input names, in the requested framework."""

    SUPPORTED_INPUT_NAMES = ()

    def supports_input(self, input_name: str) -> bool:
        """Check whether the DummyInputGenerator supports the generation of the requested input."""
        return any(input_name.startswith(supported_input_name) for supported_input_name in self.SUPPORTED_INPUT_NAMES)

    @abstractmethod
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        """Generate the dummy input matching input_name for the requested framework."""
        raise NotImplementedError

    @staticmethod
    def random_int_tensor(
        shape: list[int], max_value: int, min_value: int = 0, framework: str = "pt", dtype: str = "int64"
    ):
        """Generate a tensor of random integers in the [min_value, max_value) range."""
        if framework == "pt":
            import torch

            return torch.randint(low=min_value, high=max_value, size=shape, dtype=DtypeMapper.pt(dtype))
        else:
            import numpy as np

            return np.random.randint(min_value, high=max_value, size=shape, dtype=DtypeMapper.np(dtype))

    @staticmethod
    def random_mask_tensor(shape: list[int], padding_side: str = "right", framework: str = "pt", dtype: str = "int64"):
        """Generate a mask tensor either right or left padded."""
        shape = tuple(shape)
        mask_length = random.randint(1, shape[-1] - 1)
        if framework == "pt":
            import torch

            mask_tensor = torch.cat(
                [
                    torch.ones(*shape[:-1], shape[-1] - mask_length, dtype=DtypeMapper.pt(dtype)),
                    torch.zeros(*shape[:-1], mask_length, dtype=DtypeMapper.pt(dtype)),
                ],
                dim=-1,
            )
            if padding_side == "left":
                mask_tensor = torch.flip(mask_tensor, [-1])
        else:
            import numpy as np

            mask_tensor = np.concatenate(
                [
                    np.ones((*shape[:-1], shape[-1] - mask_length), dtype=DtypeMapper.np(dtype)),
                    np.zeros((*shape[:-1], mask_length), dtype=DtypeMapper.np(dtype)),
                ],
                axis=-1,
            )
            if padding_side == "left":
                mask_tensor = np.flip(mask_tensor, [-1])
        return mask_tensor

    @staticmethod
    def random_float_tensor(
        shape: list[int], min_value: float = 0, max_value: float = 1, framework: str = "pt", dtype: str = "fp32"
    ):
        """Generate a tensor of random floats in the [min_value, max_value) range."""
        if framework == "pt":
            import torch

            return torch.empty(shape, dtype=DtypeMapper.pt(dtype)).uniform_(min_value, max_value)
        else:
            import numpy as np

            return np.random.uniform(low=min_value, high=max_value, size=shape).astype(DtypeMapper.np(dtype))

    @staticmethod
    def constant_tensor(shape: list[int], value: float = 1, dtype: Optional[Any] = None, framework: str = "pt"):
        """Generate a constant tensor."""
        if framework == "pt":
            import torch

            return torch.full(shape, value, dtype=dtype)
        else:
            import numpy as np

            return np.full(shape, value, dtype=dtype)

    @staticmethod
    def _infer_framework_from_input(input_) -> str:
        import numpy as np
        import torch

        if isinstance(input_, np.ndarray):
            return "np"
        elif isinstance(input_, torch.Tensor):
            return "pt"
        else:
            raise RuntimeError(f"Could not infer the framework from {input_}")

    @classmethod
    def concat_inputs(cls, inputs, dim: int):
        """Concatenate inputs together."""
        if not inputs:
            raise ValueError("You did not provide any inputs to concat")
        framework = cls._infer_framework_from_input(inputs[0])
        if framework == "pt":
            import torch

            return torch.cat(inputs, dim=dim)
        else:
            import numpy as np

            return np.concatenate(inputs, axis=dim)

    @classmethod
    def pad_input_on_dim(
        cls,
        input_,
        dim: int,
        desired_length: Optional[int] = None,
        padding_length: Optional[int] = None,
        value: float = 1,
        dtype: Optional[Any] = None,
    ):
        """Pad an input either to the desired length, or by a padding length."""
        if (desired_length is None and padding_length is None) or (
            desired_length is not None and padding_length is not None
        ):
            raise ValueError("You need to provide either `desired_length` or `padding_length`")
        framework = cls._infer_framework_from_input(input_)
        shape = input_.shape
        padding_shape = list(shape)
        diff = desired_length - shape[dim] if desired_length else padding_length
        if diff <= 0:
            return input_
        padding_shape[dim] = diff
        return cls.concat_inputs(
            [input_, cls.constant_tensor(padding_shape, value=value, dtype=dtype, framework=framework)], dim=dim
        )


class DummyTextInputGenerator(DummyInputGenerator):
    """Dummy encoder text input generator."""

    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "encoder_attention_mask",
        "global_attention_mask",
        "token_type_ids",
        "position_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        random_num_choices_range: Optional[tuple[int, int]] = None,
        padding_side: str = "right",
        **kwargs,
    ):
        self.task = task
        self.vocab_size = normalized_config.vocab_size

        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length
        if random_num_choices_range:
            low, high = random_num_choices_range
            self.num_choices = random.randint(low, high)
        else:
            self.num_choices = num_choices
        self.padding_side = padding_side
        self.normalized_config = normalized_config

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        min_value = 0

        if input_name == "position_ids":
            max_value = self.sequence_length
        elif input_name == "input_ids":
            max_value = self.vocab_size
        else:
            max_value = 2

        if self.task == TaskType.MULTIPLE_CHOICE:
            shape = [self.batch_size, self.num_choices, self.sequence_length]
        else:
            shape = [self.batch_size, self.sequence_length]

        if input_name in ["attention_mask", "encoder_attention_mask"]:
            return self.random_mask_tensor(shape, padding_side=self.padding_side, framework=framework, dtype=int_dtype)
        else:
            return self.random_int_tensor(shape, max_value, min_value=min_value, framework=framework, dtype=int_dtype)


class DummyDecoderTextInputGenerator(DummyTextInputGenerator):
    """Dummy decoder text input generator."""

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
    )


class DummySeq2SeqDecoderTextInputGenerator(DummyDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        random_num_choices_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
        )
        self.hidden_size = normalized_config.hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["encoder_outputs", "encoder_hidden_states"]:
            return (
                self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.hidden_size],
                    min_value=0,
                    max_value=1,
                    framework=framework,
                    dtype=float_dtype,
                ),
                None,
                None,
            )

        return super().generate(input_name, framework=framework, int_dtype=int_dtype)


class DummyPastKeyValuesGenerator(DummyInputGenerator):
    """Dummy past_key_values input generator."""

    SUPPORTED_INPUT_NAMES = ("past_key_values",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        self.num_layers = normalized_config.num_layers
        self.num_attention_heads = normalized_config.num_attention_heads
        self.hidden_size = normalized_config.hidden_size
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class DummySeq2SeqPastKeyValuesGenerator(DummyInputGenerator):
    """Dummy past_key_values input generator for seq2seq architectures."""

    SUPPORTED_INPUT_NAMES = ("past_key_values", "cache_position")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        encoder_sequence_length: Optional[int] = None,
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        self.normalized_config = normalized_config
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length
        self.encoder_sequence_length = (
            self.sequence_length if encoder_sequence_length is None else encoder_sequence_length
        )

        self.encoder_num_attention_heads = self.normalized_config.encoder_num_attention_heads
        self.decoder_num_attention_heads = self.normalized_config.decoder_num_attention_heads
        self.encoder_hidden_size = self.normalized_config.hidden_size
        self.decoder_hidden_size = self.normalized_config.hidden_size
        self.decoder_num_layers = self.normalized_config.decoder_num_layers

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "past_key_values":
            encoder_shape = (
                self.batch_size,
                self.encoder_num_attention_heads,
                self.encoder_sequence_length,
                self.encoder_hidden_size // self.encoder_num_attention_heads,
            )
            decoder_shape = (
                self.batch_size,
                self.decoder_num_attention_heads,
                self.sequence_length,
                self.decoder_hidden_size // self.decoder_num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(encoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.decoder_num_layers)
            ]

        elif input_name == "cache_position":
            return self.random_int_tensor(
                shape=[1],
                max_value=self.sequence_length,
                framework=framework,
                dtype=int_dtype,
            )

        raise ValueError(f"Unsupported input name {input_name}")


class DummyBboxInputGenerator(DummyInputGenerator):
    """Dummy bbox input generator."""

    SUPPORTED_INPUT_NAMES = ("bbox",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size
        if random_sequence_length_range:
            low, high = random_sequence_length_range
            self.sequence_length = random.randint(low, high)
        else:
            self.sequence_length = sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        return self.random_int_tensor(
            [self.batch_size, self.sequence_length, 4],
            1,
            framework=framework,
            dtype=int_dtype,
        )


class DummyVisionInputGenerator(DummyInputGenerator):
    """Dummy vision input generator."""

    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
        "visual_embeds",
        "visual_token_type_ids",
        "visual_attention_mask",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        visual_seq_length: int = DEFAULT_DUMMY_SHAPES["visual_seq_length"],
        **kwargs,
    ):
        self.task = task

        # Some vision models can take any input sizes, in this case we use the values provided as parameters.
        if normalized_config.has_attribute("num_channels"):
            self.num_channels = normalized_config.num_channels
        else:
            self.num_channels = num_channels

        if normalized_config.has_attribute("image_size"):
            self.image_size = normalized_config.image_size
        elif normalized_config.has_attribute("input_size"):
            input_size = normalized_config.input_size
            self.num_channels = input_size[0]
            self.image_size = input_size[1:]
        else:
            self.image_size = (height, width)

        if not isinstance(self.image_size, (tuple, list)):
            self.image_size = (self.image_size, self.image_size)
        self.batch_size = batch_size
        self.height, self.width = self.image_size
        self.visual_seq_length = visual_seq_length
        self.visual_embedding_dim = getattr(normalized_config, "visual_embedding_dim", 512)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "pixel_mask":
            return self.random_int_tensor(
                shape=[self.batch_size, self.height, self.width],
                max_value=1,
                framework=framework,
                dtype=int_dtype,
            )
        elif input_name in "visual_attention_mask":
            return self.random_mask_tensor(
                shape=[self.batch_size, self.visual_seq_length],
                padding_side="right",
                framework=framework,
                dtype=int_dtype,
            )

        elif input_name == "visual_token_type_ids":
            return self.random_int_tensor(
                shape=[self.batch_size, self.visual_seq_length],
                max_value=1,
                framework=framework,
                dtype=int_dtype,
            )

        elif input_name == "visual_embeds":
            return self.random_float_tensor(
                shape=[self.batch_size, self.visual_seq_length, self.visual_embedding_dim],
                framework=framework,
                dtype=float_dtype,
            )
        else:
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_channels, self.height, self.width],
                framework=framework,
                dtype=float_dtype,
            )


class DummyAudioInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input_features", "input_values")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        feature_size: int = DEFAULT_DUMMY_SHAPES["feature_size"],
        nb_max_frames: int = DEFAULT_DUMMY_SHAPES["nb_max_frames"],
        audio_sequence_length: int = DEFAULT_DUMMY_SHAPES["audio_sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.normalized_config = normalized_config

        if hasattr(self.normalized_config, "feature_size"):
            self.feature_size = self.normalized_config.feature_size
        else:
            self.feature_size = feature_size
        self.nb_max_frames = nb_max_frames
        self.batch_size = batch_size
        self.sequence_length = audio_sequence_length

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_values":  # raw waveform
            return self.random_float_tensor(
                shape=[self.batch_size, self.sequence_length],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        else:
            return self.random_float_tensor(
                shape=[self.batch_size, self.feature_size, self.nb_max_frames],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )


class MistralDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.head_dim = getattr(normalized_config, "head_dim", self.hidden_size // self.num_attention_heads)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class GemmaDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.head_dim = normalized_config.head_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class BloomDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """Dummy past_key_values input generator for Bloom models.

    Bloom uses batch_first format which is the same as the base class.
    """


class FalconDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            **kwargs,
        )
        self.num_kv_heads = (
            normalized_config.num_kv_heads
            if (normalized_config.new_decoder_architecture or not normalized_config.multi_query)
            else 1
        )
        self.head_dim = self.hidden_size // self.num_attention_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_kv_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            **kwargs,
        )
        self.multi_query = normalized_config.multi_query

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_attention_heads if not self.multi_query else 1,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


# ============================================================================
# Diffusers Input Generators
# ============================================================================


class DummyTimestepInputGenerator(DummyInputGenerator):
    """Dummy timestep input generator for UNet diffusion models."""

    SUPPORTED_INPUT_NAMES = ("timestep", "timestep_cond", "text_embeds", "time_ids")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.vocab_size = getattr(normalized_config, "vocab_size", 1000)
        self.text_encoder_projection_dim = getattr(normalized_config, "text_encoder_projection_dim", None)
        self.time_ids_size = 5 if getattr(normalized_config, "requires_aesthetics_score", False) else 6
        self.time_cond_proj_dim = getattr(normalized_config, "time_cond_proj_dim", None)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        match input_name:
            case "timestep":
                # timestep is a scalar (0-dimensional tensor)
                return self.random_float_tensor(
                    shape=[],
                    max_value=self.vocab_size,
                    framework=framework,
                    dtype=float_dtype,
                )
            case "text_embeds":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.text_encoder_projection_dim],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "time_ids":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.time_ids_size],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "timestep_cond":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.time_cond_proj_dim],
                    framework=framework,
                    dtype=float_dtype,
                )
        raise ValueError(f"Unsupported input name: {input_name}")


class DummyUNetInputGenerator(DummyInputGenerator):
    """Dummy input generator for UNet models (sample, encoder_hidden_states)."""

    SUPPORTED_INPUT_NAMES = (
        "sample",
        "encoder_hidden_states",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # UNet config attributes
        self.num_channels = getattr(normalized_config, "in_channels", 4)
        self.image_size = getattr(normalized_config, "sample_size", 64)
        self.hidden_size = getattr(normalized_config, "cross_attention_dim", 768)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        match input_name:
            case "sample":
                height = self.image_size
                width = self.image_size
                if isinstance(self.image_size, (list, tuple)):
                    height, width = self.image_size
                return self.random_float_tensor(
                    shape=[self.batch_size, self.num_channels, height, width],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "encoder_hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.hidden_size],
                    framework=framework,
                    dtype=float_dtype,
                )
        raise ValueError(f"Unsupported input name: {input_name}")


class DummyVaeInputGenerator(DummyInputGenerator):
    """Dummy input generator for VAE encoder/decoder."""

    SUPPORTED_INPUT_NAMES = ("sample", "latent_sample")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size

        # VAE config
        self.in_channels = getattr(normalized_config, "in_channels", 3)
        self.latent_channels = getattr(normalized_config, "latent_channels", 4)
        self.sample_size = getattr(normalized_config, "sample_size", 512)

        if isinstance(self.sample_size, int):
            self.height = self.sample_size
            self.width = self.sample_size
        else:
            self.height, self.width = self.sample_size

        # Calculate latent dimensions based on downsampling factor
        down_block_types = getattr(normalized_config, "down_block_types", None)
        if down_block_types:
            self.down_sampling_factor = 2 ** (len(down_block_types) - 1)
        else:
            self.down_sampling_factor = 8  # default for SD VAE

        self.latent_height = self.height // self.down_sampling_factor
        self.latent_width = self.width // self.down_sampling_factor

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        match input_name:
            case "sample":
                # VAE encoder input: full resolution image
                return self.random_float_tensor(
                    shape=[self.batch_size, self.in_channels, self.height, self.width],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "latent_sample":
                # VAE decoder input: latent space
                return self.random_float_tensor(
                    shape=[self.batch_size, self.latent_channels, self.latent_height, self.latent_width],
                    framework=framework,
                    dtype=float_dtype,
                )
        raise ValueError(f"Unsupported input name: {input_name}")


class DummySD3TransformerInputGenerator(DummyInputGenerator):
    """Dummy input generator for SD3 Transformer."""

    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.num_channels = getattr(normalized_config, "in_channels", 16)
        self.sample_size = getattr(normalized_config, "sample_size", 128)
        self.joint_attention_dim = getattr(normalized_config, "joint_attention_dim", 4096)
        self.pooled_projection_dim = getattr(normalized_config, "pooled_projection_dim", 2048)

        if isinstance(self.sample_size, int):
            self.height = self.sample_size
            self.width = self.sample_size
        else:
            self.height, self.width = self.sample_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        match input_name:
            case "hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.num_channels, self.height, self.width],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "encoder_hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.joint_attention_dim],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "pooled_projections":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.pooled_projection_dim],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "timestep":
                return self.random_float_tensor(
                    shape=[self.batch_size],
                    framework=framework,
                    dtype=float_dtype,
                )
        raise ValueError(f"Unsupported input name: {input_name}")


class DummyFluxTransformerInputGenerator(DummyInputGenerator):
    """Dummy input generator for Flux Transformer."""

    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
        "txt_ids",
        "img_ids",
        "guidance",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.num_channels = getattr(normalized_config, "in_channels", 64)
        self.sample_size = getattr(normalized_config, "sample_size", 128)
        self.joint_attention_dim = getattr(normalized_config, "joint_attention_dim", 4096)
        self.pooled_projection_dim = getattr(normalized_config, "pooled_projection_dim", 768)
        self.guidance_embeds = getattr(normalized_config, "guidance_embeds", False)

        if isinstance(self.sample_size, int):
            self.height = self.sample_size
            self.width = self.sample_size
        else:
            self.height, self.width = self.sample_size

        self.packed_height_width = (self.height * self.width) // 4

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        match input_name:
            case "hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.packed_height_width, self.num_channels],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "encoder_hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.joint_attention_dim],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "pooled_projections":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.pooled_projection_dim],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "timestep":
                return self.random_float_tensor(
                    shape=[self.batch_size],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "txt_ids":
                return self.random_float_tensor(
                    shape=[self.sequence_length, 3],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "img_ids":
                return self.random_float_tensor(
                    shape=[self.packed_height_width, 3],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "guidance":
                if self.guidance_embeds:
                    return self.random_float_tensor(
                        shape=[self.batch_size],
                        framework=framework,
                        dtype=float_dtype,
                    )
                return None
        raise ValueError(f"Unsupported input name: {input_name}")


class DummySanaTransformerInputGenerator(DummyInputGenerator):
    """Dummy input generator for Sana Transformer."""

    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "timestep",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.num_channels = getattr(normalized_config, "in_channels", 32)
        self.sample_size = getattr(normalized_config, "sample_size", 32)
        self.caption_channels = getattr(normalized_config, "caption_channels", 2304)

        if isinstance(self.sample_size, int):
            self.height = self.sample_size
            self.width = self.sample_size
        else:
            self.height, self.width = self.sample_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        match input_name:
            case "hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.num_channels, self.height, self.width],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "encoder_hidden_states":
                return self.random_float_tensor(
                    shape=[self.batch_size, self.sequence_length, self.caption_channels],
                    framework=framework,
                    dtype=float_dtype,
                )
            case "encoder_attention_mask":
                return self.random_mask_tensor(
                    shape=[self.batch_size, self.sequence_length],
                    framework=framework,
                    dtype=int_dtype,
                )
            case "timestep":
                return self.random_float_tensor(
                    shape=[self.batch_size],
                    framework=framework,
                    dtype=float_dtype,
                )
        raise ValueError(f"Unsupported input name: {input_name}")
