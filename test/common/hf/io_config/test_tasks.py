# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from olive.common.hf.io_config.io_resolver import is_task_supported
from olive.common.hf.io_config.tasks import TaskType, map_task_synonym


class TestIsTaskSupported:
    def test_supported_task(self):
        assert is_task_supported("text-generation") is True
        assert is_task_supported("text-classification") is True

    def test_unsupported_task(self):
        assert is_task_supported("unknown-task") is False


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
