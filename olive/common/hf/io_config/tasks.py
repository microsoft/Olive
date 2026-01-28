# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum


class TaskType(str, Enum):
    """Enum for supported task types."""

    # Audio tasks
    AUDIO_CLASSIFICATION = "audio-classification"
    AUDIO_FRAME_CLASSIFICATION = "audio-frame-classification"
    AUDIO_XVECTOR = "audio-xvector"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"

    # Image tasks
    DEPTH_ESTIMATION = "depth-estimation"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_SEGMENTATION = "image-segmentation"
    IMAGE_TO_IMAGE = "image-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    KEYPOINT_DETECTION = "keypoint-detection"
    MASK_GENERATION = "mask-generation"
    MASKED_IM = "masked-im"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    ZERO_SHOT_OBJECT_DETECTION = "zero-shot-object-detection"

    # Text tasks
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"
    FEATURE_EXTRACTION = "feature-extraction"
    FEATURE_EXTRACTION_WITH_PAST = "feature-extraction-with-past"
    FILL_MASK = "fill-mask"
    MULTIPLE_CHOICE = "multiple-choice"
    QUESTION_ANSWERING = "question-answering"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_GENERATION = "text-generation"
    TEXT_GENERATION_WITH_PAST = "text-generation-with-past"
    TEXT2TEXT_GENERATION = "text2text-generation"
    TEXT2TEXT_GENERATION_WITH_PAST = "text2text-generation-with-past"
    TIME_SERIES_FORECASTING = "time-series-forecasting"
    TOKEN_CLASSIFICATION = "token-classification"

    # Vision-language tasks
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"


# Task synonyms mapping
_TASK_SYNONYMS = {
    "default": TaskType.FEATURE_EXTRACTION,
    "masked-lm": TaskType.FILL_MASK,
    "causal-lm": TaskType.TEXT_GENERATION,
    "causal-lm-with-past": TaskType.TEXT_GENERATION_WITH_PAST,
    "seq2seq-lm": TaskType.TEXT2TEXT_GENERATION,
    "seq2seq-lm-with-past": TaskType.TEXT2TEXT_GENERATION_WITH_PAST,
    "sequence-classification": TaskType.TEXT_CLASSIFICATION,
    "speech2seq-lm": TaskType.AUTOMATIC_SPEECH_RECOGNITION,
}


def map_task_synonym(task: str) -> str:
    """Map task synonyms to canonical task names."""
    return _TASK_SYNONYMS.get(task, task)
