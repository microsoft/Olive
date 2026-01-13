# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

from olive.common.hf.io_config.base import OnnxConfig
from olive.common.hf.io_config.config import (
    AudioOnnxConfig,
    AudioToTextOnnxConfig,
    TextAndVisionOnnxConfig,
    TextDecoderOnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
    TextEncoderOnnxConfig,
    TextSeq2SeqOnnxConfig,
    VisionOnnxConfig,
)
from olive.common.hf.io_config.input_generators import (
    BloomDummyPastKeyValuesGenerator,
    DummyFluxTransformerInputGenerator,
    DummySanaTransformerInputGenerator,
    DummySD3TransformerInputGenerator,
    DummyTextInputGenerator,
    # Diffusers input generators
    DummyTimestepInputGenerator,
    DummyUNetInputGenerator,
    DummyVaeInputGenerator,
    DummyVisionInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    GemmaDummyPastKeyValuesGenerator,
    GPTBigCodeDummyPastKeyValuesGenerator,
    MistralDummyPastKeyValuesGenerator,
)
from olive.common.hf.io_config.normalized_config import (
    BartLikeNormalizedTextConfig,
    BloomNormalizedTextConfig,
    CLIPNormalizedConfig,
    GPT2LikeNormalizedTextConfig,
    GPTBigCodeNormalizedTextConfig,
    NormalizedConfig,
    NormalizedFluxTransformerConfig,
    NormalizedSanaTransformerConfig,
    NormalizedSD3TransformerConfig,
    NormalizedTextConfig,
    NormalizedTextConfigWithGQA,
    # Diffusers normalized configs
    NormalizedUNetConfig,
    NormalizedVaeConfig,
    NormalizedVisionConfig,
    T5LikeNormalizedTextConfig,
    WhisperLikeNormalizedTextConfig,
)
from olive.common.hf.io_config.tasks import (
    COMMON_TEXT2TEXT_GENERATION_TASKS,
    COMMON_TEXT_GENERATION_TASKS,
    COMMON_TEXT_TASKS,
    TaskType,
)

# ============================================================================
# Registry for model configs
# ============================================================================
_ONNX_CONFIG_REGISTRY: dict[str, dict[str, type]] = {}


def register_onnx_config(model_type: str, *tasks: str):
    """Register an ONNX config class for a model type and tasks."""

    def decorator(cls):
        _ONNX_CONFIG_REGISTRY.setdefault(model_type, {})
        for task in tasks:
            _ONNX_CONFIG_REGISTRY[model_type][task] = cls
        return cls

    return decorator


def get_onnx_config_class(model_type: str, task: str) -> type:
    """Get the ONNX config class for a model type and task."""
    if model_type not in _ONNX_CONFIG_REGISTRY:
        raise KeyError(f"Model type '{model_type}' is not supported.")
    if task not in _ONNX_CONFIG_REGISTRY[model_type]:
        supported_tasks = list(_ONNX_CONFIG_REGISTRY[model_type].keys())
        raise KeyError(
            f"Task '{task}' is not supported for model type '{model_type}'. Supported tasks: {supported_tasks}"
        )
    return _ONNX_CONFIG_REGISTRY[model_type][task]


def get_supported_model_types() -> list[str]:
    """Get list of supported model types."""
    return list(_ONNX_CONFIG_REGISTRY.keys())


def get_supported_tasks_for_model(model_type: str) -> list[str]:
    """Get list of supported tasks for a model type."""
    if model_type not in _ONNX_CONFIG_REGISTRY:
        raise KeyError(f"Model type '{model_type}' is not supported.")
    return list(_ONNX_CONFIG_REGISTRY[model_type].keys())


# ============================================================================
# Text Encoder Models (BERT-like)
# ============================================================================


@register_onnx_config("bert", *COMMON_TEXT_TASKS)
class BertOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.task == TaskType.MULTIPLE_CHOICE:
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
        }


@register_onnx_config("visual_bert", TaskType.FEATURE_EXTRACTION)
class VisualBertOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyVisionInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "visual_embeds": {0: "batch_size", 1: "visual_seq_length", 2: "visual_embedding_dim"},
            "visual_attention_mask": {0: "batch_size", 1: "visual_seq_length"},
            "visual_token_type_ids": {0: "batch_size", 1: "visual_seq_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length + visual_seq_length"},
        }


@register_onnx_config("albert", *COMMON_TEXT_TASKS)
class AlbertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("convbert", *COMMON_TEXT_TASKS)
class ConvBertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("electra", *COMMON_TEXT_TASKS)
class ElectraOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("roformer", *COMMON_TEXT_TASKS)
class RoFormerOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("squeezebert", *COMMON_TEXT_TASKS)
class SqueezeBertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("nystromformer", *COMMON_TEXT_TASKS)
class NystromformerOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("xlm", *COMMON_TEXT_TASKS)
class XLMOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("splinter", TaskType.FEATURE_EXTRACTION, TaskType.QUESTION_ANSWERING)
class SplinterOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("rembert", *COMMON_TEXT_TASKS)
class RemBertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("megatron-bert", *COMMON_TEXT_TASKS)
class MegatronBertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("distilbert", *COMMON_TEXT_TASKS)
class DistilBertOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.task == TaskType.MULTIPLE_CHOICE:
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


@register_onnx_config(
    "modernbert",
    TaskType.FEATURE_EXTRACTION,
    TaskType.FILL_MASK,
    TaskType.TEXT_CLASSIFICATION,
    TaskType.TOKEN_CLASSIFICATION,
)
class ModernBertOnnxConfig(DistilBertOnnxConfig):
    pass


@register_onnx_config("mpnet", *COMMON_TEXT_TASKS)
class MPNetOnnxConfig(DistilBertOnnxConfig):
    pass


@register_onnx_config("roberta", *COMMON_TEXT_TASKS)
class RobertaOnnxConfig(DistilBertOnnxConfig):
    pass


@register_onnx_config("camembert", *COMMON_TEXT_TASKS)
class CamembertOnnxConfig(DistilBertOnnxConfig):
    pass


@register_onnx_config("flaubert", *COMMON_TEXT_TASKS)
class FlaubertOnnxConfig(BertOnnxConfig):
    pass


@register_onnx_config("ibert", *COMMON_TEXT_TASKS)
class IBertOnnxConfig(DistilBertOnnxConfig):
    pass


@register_onnx_config("xlm-roberta", *COMMON_TEXT_TASKS)
class XLMRobertaOnnxConfig(DistilBertOnnxConfig):
    pass


@register_onnx_config(
    "deberta",
    TaskType.FEATURE_EXTRACTION,
    TaskType.FILL_MASK,
    TaskType.TEXT_CLASSIFICATION,
    TaskType.TOKEN_CLASSIFICATION,
    TaskType.QUESTION_ANSWERING,
)
class DebertaOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop("token_type_ids")
        return common_inputs


@register_onnx_config("deberta_v2", *COMMON_TEXT_TASKS)
class DebertaV2OnnxConfig(DebertaOnnxConfig):
    pass


@register_onnx_config(
    "esm", TaskType.FEATURE_EXTRACTION, TaskType.FILL_MASK, TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION
)
class EsmOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
        }


# ============================================================================
# Text Decoder Models (GPT-like)
# ============================================================================


@register_onnx_config(
    "gpt2", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]
)
class GPT2OnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = GPT2LikeNormalizedTextConfig


@register_onnx_config(
    "gptj", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.QUESTION_ANSWERING]
)
class GPTJOnnxConfig(GPT2OnnxConfig):
    pass


@register_onnx_config("codegen", *COMMON_TEXT_GENERATION_TASKS)
class CodeGenOnnxConfig(GPT2OnnxConfig):
    pass


@register_onnx_config("imagegpt", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class ImageGPTOnnxConfig(GPT2OnnxConfig):
    pass


@register_onnx_config("gpt_neo", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class GPTNeoOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads="num_heads")


@register_onnx_config("gpt_neox", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class GPTNeoXOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_onnx_config(
    "opt", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.QUESTION_ANSWERING]
)
class OPTOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_onnx_config("llama", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class LlamaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_onnx_config("arcee", *COMMON_TEXT_GENERATION_TASKS)
class ArceeOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("cohere", *COMMON_TEXT_GENERATION_TASKS)
class CohereOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_onnx_config("glm", *COMMON_TEXT_GENERATION_TASKS)
class GLMOnnxConfig(LlamaOnnxConfig):
    pass


@register_onnx_config("helium", *COMMON_TEXT_GENERATION_TASKS)
class HeliumOnnxConfig(LlamaOnnxConfig):
    pass


@register_onnx_config("smollm3", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class SmolLM3OnnxConfig(LlamaOnnxConfig):
    pass


@register_onnx_config("stablelm", *COMMON_TEXT_GENERATION_TASKS)
class StableLMOnnxConfig(LlamaOnnxConfig):
    pass


@register_onnx_config("olmo", *COMMON_TEXT_GENERATION_TASKS)
class OlmoOnnxConfig(LlamaOnnxConfig):
    pass


@register_onnx_config("olmo2", *COMMON_TEXT_GENERATION_TASKS)
class Olmo2OnnxConfig(OlmoOnnxConfig):
    pass


@register_onnx_config(
    "qwen2", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]
)
class Qwen2OnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("qwen3", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class Qwen3OnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config(
    "qwen3_moe", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]
)
class Qwen3MoeOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("mistral", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class MistralOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("mixtral", *COMMON_TEXT_GENERATION_TASKS)
class MixtralOnnxConfig(MistralOnnxConfig):
    pass


@register_onnx_config("phi", *COMMON_TEXT_GENERATION_TASKS)
class PhiOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_onnx_config("phi3", *COMMON_TEXT_GENERATION_TASKS)
class Phi3OnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("gemma", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class GemmaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("gemma2", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class Gemma2OnnxConfig(GemmaOnnxConfig):
    pass


@register_onnx_config("gemma3", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class Gemma3OnnxConfig(GemmaOnnxConfig):
    pass


@register_onnx_config("granite", *COMMON_TEXT_GENERATION_TASKS)
class GraniteOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_onnx_config("bloom", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class BloomOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, BloomDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = BloomDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = BloomNormalizedTextConfig


@register_onnx_config(
    "falcon", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.QUESTION_ANSWERING]
)
class FalconOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, FalconDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = FalconDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_onnx_config("mpt", *COMMON_TEXT_GENERATION_TASKS)
class MPTOnnxConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="n_heads", hidden_size="d_model", num_layers="n_layers"
    )


@register_onnx_config("gpt_bigcode", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class GPTBigCodeOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GPTBigCodeDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = GPTBigCodeNormalizedTextConfig


@register_onnx_config("starcoder2", *[*COMMON_TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION])
class Starcoder2OnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


# ============================================================================
# Seq2Seq Models (T5-like, BART-like)
# ============================================================================


@register_onnx_config("t5", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class T5OnnxConfig(TextSeq2SeqOnnxConfig):
    NORMALIZED_CONFIG_CLASS = T5LikeNormalizedTextConfig


@register_onnx_config("mt5", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class MT5OnnxConfig(T5OnnxConfig):
    pass


@register_onnx_config("longt5", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class LongT5OnnxConfig(T5OnnxConfig):
    pass


@register_onnx_config(
    "bart", *[*COMMON_TEXT2TEXT_GENERATION_TASKS, TaskType.TEXT_CLASSIFICATION, TaskType.QUESTION_ANSWERING]
)
class BartOnnxConfig(TextSeq2SeqOnnxConfig):
    NORMALIZED_CONFIG_CLASS = BartLikeNormalizedTextConfig


@register_onnx_config("mbart", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class MBartOnnxConfig(BartOnnxConfig):
    pass


@register_onnx_config("blenderbot", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class BlenderbotOnnxConfig(BartOnnxConfig):
    pass


@register_onnx_config("blenderbot_small", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class BlenderbotSmallOnnxConfig(BartOnnxConfig):
    pass


@register_onnx_config("pegasus", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class PegasusOnnxConfig(BartOnnxConfig):
    pass


@register_onnx_config("marian", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class MarianOnnxConfig(BartOnnxConfig):
    pass


@register_onnx_config("m2m_100", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class M2M100OnnxConfig(BartOnnxConfig):
    pass


# ============================================================================
# Vision Models
# ============================================================================


@register_onnx_config("vit", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class ViTOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}


@register_onnx_config("deit", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class DeiTOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("beit", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class BeitOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("swin", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class SwinOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("dinov2", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class Dinov2OnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("resnet", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class ResNetOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}


@register_onnx_config("convnext", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class ConvNextOnnxConfig(ResNetOnnxConfig):
    pass


@register_onnx_config("convnextv2", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class ConvNextV2OnnxConfig(ResNetOnnxConfig):
    pass


@register_onnx_config("poolformer", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class PoolFormerOnnxConfig(ResNetOnnxConfig):
    pass


@register_onnx_config("regnet", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class RegNetOnnxConfig(ResNetOnnxConfig):
    pass


@register_onnx_config("mobilenet_v1", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class MobileNetV1OnnxConfig(ResNetOnnxConfig):
    pass


@register_onnx_config("mobilenet_v2", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class MobileNetV2OnnxConfig(ResNetOnnxConfig):
    pass


@register_onnx_config("mobilevit", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class MobileViTOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("levit", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class LevitOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config(
    "segformer", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION, TaskType.SEMANTIC_SEGMENTATION
)
class SegformerOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("cvt", TaskType.FEATURE_EXTRACTION, TaskType.IMAGE_CLASSIFICATION)
class CvtOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("yolos", TaskType.FEATURE_EXTRACTION, TaskType.OBJECT_DETECTION)
class YolosOnnxConfig(ViTOnnxConfig):
    pass


@register_onnx_config("detr", TaskType.FEATURE_EXTRACTION, TaskType.OBJECT_DETECTION)
class DetrOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "pixel_mask": {0: "batch_size", 1: "height", 2: "width"},
        }


@register_onnx_config("table-transformer", TaskType.FEATURE_EXTRACTION, TaskType.OBJECT_DETECTION)
class TableTransformerOnnxConfig(DetrOnnxConfig):
    pass


# ============================================================================
# Audio Models
# ============================================================================


@register_onnx_config("whisper", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class WhisperOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = WhisperLikeNormalizedTextConfig


@register_onnx_config(
    "wav2vec2",
    TaskType.FEATURE_EXTRACTION,
    TaskType.AUTOMATIC_SPEECH_RECOGNITION,
    TaskType.AUDIO_CLASSIFICATION,
    TaskType.AUDIO_FRAME_CLASSIFICATION,
)
class Wav2Vec2OnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig


@register_onnx_config(
    "hubert", TaskType.FEATURE_EXTRACTION, TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION
)
class HubertOnnxConfig(Wav2Vec2OnnxConfig):
    pass


@register_onnx_config(
    "wavlm",
    TaskType.FEATURE_EXTRACTION,
    TaskType.AUTOMATIC_SPEECH_RECOGNITION,
    TaskType.AUDIO_CLASSIFICATION,
    TaskType.AUDIO_FRAME_CLASSIFICATION,
    TaskType.AUDIO_XVECTOR,
)
class WavLMOnnxConfig(Wav2Vec2OnnxConfig):
    pass


@register_onnx_config(
    "sew", TaskType.FEATURE_EXTRACTION, TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION
)
class SEWOnnxConfig(Wav2Vec2OnnxConfig):
    pass


@register_onnx_config(
    "sew_d", TaskType.FEATURE_EXTRACTION, TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION
)
class SEWDOnnxConfig(Wav2Vec2OnnxConfig):
    pass


@register_onnx_config(
    "unispeech", TaskType.FEATURE_EXTRACTION, TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION
)
class UniSpeechOnnxConfig(Wav2Vec2OnnxConfig):
    pass


@register_onnx_config(
    "unispeech_sat",
    TaskType.FEATURE_EXTRACTION,
    TaskType.AUTOMATIC_SPEECH_RECOGNITION,
    TaskType.AUDIO_CLASSIFICATION,
    TaskType.AUDIO_FRAME_CLASSIFICATION,
    TaskType.AUDIO_XVECTOR,
)
class UniSpeechSATOnnxConfig(Wav2Vec2OnnxConfig):
    pass


# ============================================================================
# Multimodal Models
# ============================================================================


@register_onnx_config("clip", TaskType.FEATURE_EXTRACTION, TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION)
class CLIPOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }


@register_onnx_config("siglip", TaskType.FEATURE_EXTRACTION, TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION)
class SiglipOnnxConfig(CLIPOnnxConfig):
    pass


@register_onnx_config("blip", TaskType.FEATURE_EXTRACTION)
class BlipOnnxConfig(CLIPOnnxConfig):
    pass


@register_onnx_config("blip_2", TaskType.FEATURE_EXTRACTION)
class Blip2OnnxConfig(CLIPOnnxConfig):
    pass


# ============================================================================
# Diffusers Models
# ============================================================================


class DiffusersTextEncoderOnnxConfig(OnnxConfig):
    """ONNX config for CLIP text encoder used in diffusers pipelines."""

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }


class DiffusersTextEncoderWithProjectionOnnxConfig(DiffusersTextEncoderOnnxConfig):
    """ONNX config for CLIP text encoder with projection (SDXL text_encoder_2)."""

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "text_embeds": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }


class DiffusersT5EncoderOnnxConfig(OnnxConfig):
    """ONNX config for T5 encoder used in SD3/Flux text_encoder_3."""

    NORMALIZED_CONFIG_CLASS = T5LikeNormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }


class UNetOnnxConfig(OnnxConfig):
    """ONNX config for UNet2DConditionModel (SD1.5, SDXL)."""

    NORMALIZED_CONFIG_CLASS = NormalizedUNetConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyUNetInputGenerator,
        DummyTimestepInputGenerator,
    )

    def __init__(self, config, task: str = "semantic-segmentation", **kwargs):
        super().__init__(config, task=task, **kwargs)
        self.is_sdxl = getattr(config, "addition_embed_type", None) == "text_time"

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "timestep": {},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }
        # SDXL specific inputs
        if self.is_sdxl:
            common_inputs["text_embeds"] = {0: "batch_size"}
            common_inputs["time_ids"] = {0: "batch_size"}

        # Optional timestep_cond
        if getattr(self._config, "time_cond_proj_dim", None):
            common_inputs["timestep_cond"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"out_sample": {0: "batch_size", 2: "height", 3: "width"}}


class VaeEncoderOnnxConfig(OnnxConfig):
    """ONNX config for VAE encoder."""

    NORMALIZED_CONFIG_CLASS = NormalizedVaeConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVaeInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"sample": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"latent_parameters": {0: "batch_size", 2: "height_latent", 3: "width_latent"}}


class VaeDecoderOnnxConfig(OnnxConfig):
    """ONNX config for VAE decoder."""

    NORMALIZED_CONFIG_CLASS = NormalizedVaeConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVaeInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"latent_sample": {0: "batch_size", 2: "height_latent", 3: "width_latent"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"sample": {0: "batch_size", 2: "height", 3: "width"}}


class SD3TransformerOnnxConfig(OnnxConfig):
    """ONNX config for SD3 Transformer."""

    NORMALIZED_CONFIG_CLASS = NormalizedSD3TransformerConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummySD3TransformerInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "hidden_states": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "batch_size"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"out_sample": {0: "batch_size", 2: "height", 3: "width"}}


class FluxTransformerOnnxConfig(OnnxConfig):
    """ONNX config for Flux Transformer."""

    NORMALIZED_CONFIG_CLASS = NormalizedFluxTransformerConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyFluxTransformerInputGenerator,)

    def __init__(self, config, task: str = "semantic-segmentation", **kwargs):
        super().__init__(config, task=task, **kwargs)
        self.guidance_embeds = getattr(config, "guidance_embeds", False)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {
            "hidden_states": {0: "batch_size", 1: "packed_height_width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "txt_ids": {0: "sequence_length"},
            "img_ids": {0: "packed_height_width"},
        }
        if self.guidance_embeds:
            common_inputs["guidance"] = {0: "batch_size"}
        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"out_sample": {0: "batch_size", 1: "packed_height_width"}}


class SanaTransformerOnnxConfig(OnnxConfig):
    """ONNX config for Sana Transformer."""

    NORMALIZED_CONFIG_CLASS = NormalizedSanaTransformerConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummySanaTransformerInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "hidden_states": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "sequence_length"},
            "timestep": {0: "batch_size"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"out_sample": {0: "batch_size", 2: "height", 3: "width"}}


class DiffusersGemma2TextEncoderOnnxConfig(OnnxConfig):
    """ONNX config for Gemma2 text encoder used in Sana."""

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}


class DcaeEncoderOnnxConfig(OnnxConfig):
    """ONNX config for DC-AE encoder used in Sana."""

    NORMALIZED_CONFIG_CLASS = NormalizedVaeConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVaeInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"sample": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"latent": {0: "batch_size", 2: "height_latent", 3: "width_latent"}}


class DcaeDecoderOnnxConfig(OnnxConfig):
    """ONNX config for DC-AE decoder used in Sana."""

    NORMALIZED_CONFIG_CLASS = NormalizedVaeConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVaeInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"latent_sample": {0: "batch_size", 2: "height_latent", 3: "width_latent"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {"sample": {0: "batch_size", 2: "height", 3: "width"}}
