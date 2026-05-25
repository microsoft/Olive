# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------
# pylint: disable=protected-access

import builtins
from unittest.mock import MagicMock, patch

import pytest

from olive.model import QairtModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.qairt.pipeline import QairtPipelinePass


@pytest.fixture(name="mock_pipeline_modules")
def mock_pipeline_modules_fixture():
    """Mock qairt and the LLMPipeline API."""
    mock_qairt = MagicMock()
    mock_recipe_cls = MagicMock()
    mock_pipeline_cls = MagicMock()

    mock_pipeline = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

    with (
        patch.dict("sys.modules", {"qairt": mock_qairt}),
        patch(
            "qairt.experimental.pipeline.torch.common.recipe.Recipe",
            mock_recipe_cls,
            create=True,
        ),
        patch(
            "qairt.experimental.pipeline.torch.llm.pipeline.LLMPipeline",
            mock_pipeline_cls,
            create=True,
        ),
        patch.dict(
            "sys.modules",
            {
                "qairt.experimental.pipeline.torch.common.recipe": MagicMock(Recipe=mock_recipe_cls),
                "qairt.experimental.pipeline.torch.llm.pipeline": MagicMock(LLMPipeline=mock_pipeline_cls),
            },
        ),
    ):
        yield {
            "qairt": mock_qairt,
            "Recipe": mock_recipe_cls,
            "LLMPipeline": mock_pipeline_cls,
            "pipeline": mock_pipeline,
        }


@pytest.fixture(name="recipe_file")
def recipe_file_fixture(tmp_path):
    # Content is irrelevant — Recipe.from_file is mocked in every test that uses this fixture.
    # The file must exist so that the recipe_path.exists() guard in _run_for_config passes.
    path = tmp_path / "recipe.yaml"
    path.write_text("")
    return path


@pytest.fixture(name="recipe_file_with_model_id")
def recipe_file_with_model_id_fixture(tmp_path):
    # Content is irrelevant — Recipe.from_file is mocked in every test that uses this fixture.
    # The file must exist so that the recipe_path.exists() guard in _run_for_config passes.
    path = tmp_path / "recipe_with_model.yaml"
    path.write_text("")
    return path


def test_pipeline_pass_default_config(mock_accelerator_spec):
    """Test that the default config has the expected parameters."""
    config = QairtPipelinePass._default_config(mock_accelerator_spec)

    assert "recipe" in config
    assert config["recipe"].required is True
    assert "cache_dir" in config
    assert config["cache_dir"].default_value is None
    assert "log_level" in config
    assert config["log_level"].default_value is None


def test_pipeline_pass_success(tmp_path, mock_hf_model, recipe_file, mock_pipeline_modules):
    """Test successful pass execution with no model_id_or_path in recipe."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {
        "cache_dir": "./pipeline_cache",
        "backend": "HTP",
        "stages": {},
    }

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(recipe_file)},
        disable_search=True,
    )

    result = pipeline_pass.run(mock_hf_model, str(output_path))

    assert isinstance(result, QairtModelHandler)
    assert result.model_path == str(output_path)
    mock_pipeline_modules["LLMPipeline"].from_pretrained.assert_called_once_with(
        mock_hf_model.model_path,
        recipe={"cache_dir": "./pipeline_cache", "backend": "HTP", "stages": {}},
    )
    mock_pipeline_modules["pipeline"].construct.assert_called_once()
    mock_pipeline_modules["pipeline"].export.assert_called_once_with(str(output_path))


def test_pipeline_pass_recipe_model_id_matches_handler(
    tmp_path, mock_hf_model, recipe_file_with_model_id, mock_pipeline_modules
):
    """Test that no error is raised when recipe model_id_or_path matches the handler path."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {
        "model_id_or_path": mock_hf_model.model_path,
        "stages": {},
    }

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(recipe_file_with_model_id)},
        disable_search=True,
    )

    result = pipeline_pass.run(mock_hf_model, str(output_path))
    assert isinstance(result, QairtModelHandler)


def test_pipeline_pass_recipe_model_id_conflict_raises(
    tmp_path, mock_hf_model, recipe_file_with_model_id, mock_pipeline_modules
):
    """Test that a ValueError is raised when recipe model_id_or_path conflicts with handler path."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {
        "model_id_or_path": "meta-llama/Llama-3.2-3B-Instruct",
        "stages": {},
    }

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(recipe_file_with_model_id)},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="Conflict between recipe model_id_or_path"):
        pipeline_pass.run(mock_hf_model, str(output_path))


def test_pipeline_pass_cache_dir_override(tmp_path, mock_hf_model, recipe_file, mock_pipeline_modules):
    """Test that Olive-level cache_dir overrides the recipe's cache_dir."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {
        "cache_dir": "./recipe_cache",
        "stages": {},
    }

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(recipe_file), "cache_dir": "/custom/cache"},
        disable_search=True,
    )

    pipeline_pass.run(mock_hf_model, str(output_path))

    call_kwargs = mock_pipeline_modules["LLMPipeline"].from_pretrained.call_args
    recipe_arg = call_kwargs.kwargs["recipe"]
    assert recipe_arg["cache_dir"] == "/custom/cache"


def test_pipeline_pass_log_level_override(tmp_path, mock_hf_model, recipe_file, mock_pipeline_modules):
    """Test that Olive-level log_level overrides the recipe's log_level."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {
        "log_level": "warn",
        "stages": {},
    }

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(recipe_file), "log_level": "DEBUG"},
        disable_search=True,
    )

    pipeline_pass.run(mock_hf_model, str(output_path))

    call_kwargs = mock_pipeline_modules["LLMPipeline"].from_pretrained.call_args
    recipe_arg = call_kwargs.kwargs["recipe"]
    assert recipe_arg["log_level"] == "DEBUG"


def test_pipeline_pass_invalid_input_model(tmp_path, mock_qairt_prepared_model, recipe_file, mock_pipeline_modules):
    """Test that ValueError is raised when input is not HfModelHandler."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {"stages": {}}

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(recipe_file)},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="QairtPipelinePass requires HfModelHandler"):
        pipeline_pass.run(mock_qairt_prepared_model, str(output_path))


def test_pipeline_pass_missing_recipe_file(tmp_path, mock_hf_model, mock_pipeline_modules):
    """Test that ValueError is raised when recipe file does not exist."""
    output_path = tmp_path / "output"

    mock_pipeline_modules["Recipe"].from_file.return_value = {"stages": {}}

    pipeline_pass = create_pass_from_dict(
        QairtPipelinePass,
        {"recipe": str(tmp_path / "nonexistent_recipe.yaml")},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="Recipe file not found"):
        pipeline_pass.run(mock_hf_model, str(output_path))


def test_pipeline_pass_import_error(tmp_path, mock_hf_model, recipe_file):
    """Test that ImportError is raised if qairt cannot be imported."""

    def import_side_effect(name, *args, **kwargs):
        if "qairt" in name:
            raise ImportError("Mock import error")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__

    with patch("builtins.__import__", side_effect=import_side_effect):
        pipeline_pass = create_pass_from_dict(
            QairtPipelinePass,
            {"recipe": str(recipe_file)},
            disable_search=True,
        )

        with pytest.raises(ImportError, match="Failed to import QAIRT Pipeline API"):
            pipeline_pass.run(mock_hf_model, str(tmp_path / "output"))
