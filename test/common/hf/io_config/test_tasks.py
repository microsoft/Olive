# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

from olive.common.hf.io_config.tasks import (
    COMMON_TEXT2TEXT_GENERATION_TASKS,
    COMMON_TEXT_GENERATION_TASKS,
    COMMON_TEXT_TASKS,
    TaskType,
    map_task_synonym,
)


class TestTaskType:
    def test_task_type_values(self):
        """Test that TaskType enum values are correct strings."""
        assert TaskType.FEATURE_EXTRACTION == "feature-extraction"
        assert TaskType.TEXT_GENERATION == "text-generation"
        assert TaskType.TEXT_CLASSIFICATION == "text-classification"
        assert TaskType.AUTOMATIC_SPEECH_RECOGNITION == "automatic-speech-recognition"
        assert TaskType.IMAGE_CLASSIFICATION == "image-classification"

    def test_task_type_string_comparison(self):
        """Test that TaskType can be compared with strings."""
        assert TaskType.FEATURE_EXTRACTION == "feature-extraction"
        assert TaskType.TEXT_GENERATION == "text-generation"


class TestTaskGroups:
    def test_common_text_tasks_contains_expected(self):
        """Test COMMON_TEXT_TASKS contains expected task types."""
        assert TaskType.FEATURE_EXTRACTION in COMMON_TEXT_TASKS
        assert TaskType.FILL_MASK in COMMON_TEXT_TASKS
        assert TaskType.TEXT_CLASSIFICATION in COMMON_TEXT_TASKS
        assert TaskType.TOKEN_CLASSIFICATION in COMMON_TEXT_TASKS

    def test_common_text_generation_tasks_contains_expected(self):
        """Test COMMON_TEXT_GENERATION_TASKS contains expected task types."""
        assert TaskType.TEXT_GENERATION in COMMON_TEXT_GENERATION_TASKS
        assert TaskType.TEXT_GENERATION_WITH_PAST in COMMON_TEXT_GENERATION_TASKS
        assert TaskType.FEATURE_EXTRACTION in COMMON_TEXT_GENERATION_TASKS

    def test_common_text2text_generation_tasks_contains_text_gen(self):
        """Test COMMON_TEXT2TEXT_GENERATION_TASKS includes text generation tasks."""
        for task in COMMON_TEXT_GENERATION_TASKS:
            assert task in COMMON_TEXT2TEXT_GENERATION_TASKS
        assert TaskType.TEXT2TEXT_GENERATION in COMMON_TEXT2TEXT_GENERATION_TASKS


class TestMapTaskSynonym:
    def test_map_known_synonyms(self):
        """Test that known task synonyms are mapped correctly."""
        assert map_task_synonym("default") == TaskType.FEATURE_EXTRACTION
        assert map_task_synonym("masked-lm") == TaskType.FILL_MASK
        assert map_task_synonym("causal-lm") == TaskType.TEXT_GENERATION
        assert map_task_synonym("seq2seq-lm") == TaskType.TEXT2TEXT_GENERATION
        assert map_task_synonym("sequence-classification") == TaskType.TEXT_CLASSIFICATION

    def test_map_unknown_task_returns_as_is(self):
        """Test that unknown tasks are returned as-is."""
        assert map_task_synonym("unknown-task") == "unknown-task"
        assert map_task_synonym("feature-extraction") == "feature-extraction"
        assert map_task_synonym("text-generation") == "text-generation"
