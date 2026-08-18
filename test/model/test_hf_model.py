# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import ANY, patch

import huggingface_hub
import pytest
import torch
import transformers

from olive.model.config.io_config import IoConfig
from olive.model.handler.hf import HfModelHandler


# pylint: disable=attribute-defined-outside-init
class TestHFModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-generation-with-past"
        self.model_name = "katuni4ka/tiny-random-phi3"
        self.revision = "585361abfee667f3c63f8b2dc4ad58405c4e34e2"

        self.local_path = huggingface_hub.snapshot_download(self.model_name, revision=self.revision)

    @pytest.mark.parametrize("local", [True, False])
    def test_load_model(self, local):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"revision": self.revision},
        )

        pytorch_model = olive_model.load_model()
        actual_class_path = f"{pytorch_model.__module__}.{pytorch_model.__class__.__name__}"
        assert actual_class_path == "transformers.models.phi3.modeling_phi3.Phi3ForCausalLM"

    @pytest.mark.parametrize("local", [True, False])
    def test_load_model_with_kwargs(self, local):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"torch_dtype": "float16"},
        )
        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.Phi3ForCausalLM)
        assert pytorch_model.dtype == torch.float16

    @pytest.mark.parametrize("local", [True, False])
    def test_model_name_or_path(self, local):
        olive_model = HfModelHandler(model_path=self.local_path if local else self.model_name, task=self.task)
        assert olive_model.model_name_or_path == str(Path(self.local_path).resolve()) if local else self.model_name

    @pytest.mark.parametrize("local", [True, False])
    @pytest.mark.parametrize("trust_remote_code", [True, False])
    @pytest.mark.parametrize("tokenizer_exists", [True, False])
    def test_save_metadata(self, local, trust_remote_code, tokenizer_exists, tmp_path):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"trust_remote_code": trust_remote_code, "revision": self.revision},
        )
        if tokenizer_exists:
            olive_model.get_hf_tokenizer().save_pretrained(tmp_path)
        saved_filepaths = olive_model.save_metadata(tmp_path)
        # transformers>=5.0.0
        assert len(saved_filepaths) == (4 if tokenizer_exists else 7)
        assert all(Path(fp).exists() for fp in saved_filepaths)
        assert isinstance(transformers.AutoConfig.from_pretrained(tmp_path), transformers.Phi3Config)
        assert isinstance(transformers.AutoTokenizer.from_pretrained(tmp_path), transformers.PreTrainedTokenizerBase)

    @pytest.mark.parametrize("local", [True, False])
    def test_save_pretrained_metadata(self, local, tmp_path):
        olive_model = HfModelHandler(
            model_path=self.local_path if local else self.model_name,
            task=self.task,
            load_kwargs={"revision": self.revision},
        )

        # modify the config and save the model
        loaded_model = olive_model.load_model()
        loaded_model.config.dummy_key = "dummy_value"
        loaded_model.save_pretrained(tmp_path)

        saved_filepaths = olive_model.save_metadata(tmp_path)
        # generation config is also saved, transformers>=5.0.0
        assert len(saved_filepaths) == 6

        with open(tmp_path / "config.json") as f:
            config = json.load(f)
        # encure already saved config is used
        assert config["dummy_key"] == "dummy_value"
        # ensure the auto_map is updated
        assert config["auto_map"] == {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
        }


def test_save_metadata_saves_processor_for_vl_model(tmp_path):
    """save_metadata should save preprocessor_config.json for multimodal models.

    Uses a mocked AutoProcessor since there's no lightweight real VL checkpoint fixture
    available; verifies the processor.save_pretrained output is captured and the
    tokenizer-like AutoProcessor return value (text-only models) is correctly skipped.
    """
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        task="text-generation-with-past",
        load_kwargs={"revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
    )

    processor_config_path = tmp_path / "preprocessor_config.json"

    class FakeProcessor:
        def save_pretrained(self, output_dir, **kwargs):
            path = Path(output_dir) / "preprocessor_config.json"
            path.write_text("{}")
            return [str(path)]

    with patch("transformers.AutoProcessor.from_pretrained", return_value=FakeProcessor()):
        saved_filepaths = olive_model.save_metadata(tmp_path)

    assert str(processor_config_path) in saved_filepaths
    assert processor_config_path.exists()


def test_save_metadata_processor_fills_in_missing_files_without_overwriting_existing(tmp_path):
    """A processor with some files already saved must still be filled in without clobbering existing ones.

    save_metadata should still look for processor files even if one (e.g. preprocessor_config.json)
    already exists, since a processor can emit several files and an earlier step may have saved only
    some of them -- but it must never overwrite a file that's already there.
    """
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        task="text-generation-with-past",
        load_kwargs={"revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "preprocessor_config.json").write_text('{"existing": true}')

    class FakeProcessor:
        def save_pretrained(self, output_dir, **kwargs):
            out = Path(output_dir)
            (out / "preprocessor_config.json").write_text('{"existing": false}')
            (out / "chat_template.json").write_text("{}")
            return [str(out / "preprocessor_config.json"), str(out / "chat_template.json")]

    with patch("transformers.AutoProcessor.from_pretrained", return_value=FakeProcessor()) as mock_from_pretrained:
        saved_filepaths = olive_model.save_metadata(tmp_path)

    mock_from_pretrained.assert_called_once()
    # the pre-existing preprocessor_config.json is untouched, but the missing chat_template.json is filled in
    assert (tmp_path / "preprocessor_config.json").read_text() == '{"existing": true}'
    assert (tmp_path / "chat_template.json").exists()
    assert str(tmp_path / "chat_template.json") in saved_filepaths
    assert str(tmp_path / "preprocessor_config.json") not in saved_filepaths


@contextmanager
def capture_warnings(logger_name: str):
    """Collect warning-or-worse records from ``logger_name`` (Olive loggers don't propagate)."""
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _Handler(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_save_metadata_warns_on_unexpected_processor_failure(tmp_path):
    """An unexpected AutoProcessor failure must be visible, not silently swallowed at debug level."""
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        task="text-generation-with-past",
        load_kwargs={"revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
    )

    with (
        patch("transformers.AutoProcessor.from_pretrained", side_effect=RuntimeError("boom")),
        capture_warnings("olive.model.handler.mixin.hf") as warnings,
    ):
        saved_filepaths = olive_model.save_metadata(tmp_path)

    assert any("boom" in message for message in warnings)
    assert not (tmp_path / "preprocessor_config.json").exists()
    # the rest of the metadata is still saved
    assert str(tmp_path / "config.json") in saved_filepaths


def test_save_metadata_skips_tokenizer_return_silently(tmp_path):
    """Text-only models: AutoProcessor returns the tokenizer -- expected, so no warning."""
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        task="text-generation-with-past",
        load_kwargs={"revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "katuni4ka/tiny-random-phi3", revision="585361abfee667f3c63f8b2dc4ad58405c4e34e2"
    )
    with (
        patch("transformers.AutoProcessor.from_pretrained", return_value=tokenizer),
        capture_warnings("olive.model.handler.mixin.hf") as warnings,
    ):
        olive_model.save_metadata(tmp_path)

    assert not warnings
    assert not (tmp_path / "preprocessor_config.json").exists()


def test_save_metadata_processor_does_not_overwrite_custom_tokenizer(tmp_path):
    """A processor save must not clobber a tokenizer an earlier step customized and saved."""
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        task="text-generation-with-past",
        load_kwargs={"revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
    )

    custom_tokenizer_config = '{"custom": true}'
    (tmp_path / "tokenizer_config.json").write_text(custom_tokenizer_config)

    class FakeProcessor:
        """Mimics ProcessorMixin.save_pretrained, which also re-saves the tokenizer."""

        def save_pretrained(self, output_dir, **kwargs):
            out = Path(output_dir)
            (out / "preprocessor_config.json").write_text('{"image_processor": true}')
            (out / "tokenizer_config.json").write_text('{"custom": false}')
            (out / "tokenizer.json").write_text("{}")
            return [str(out / "preprocessor_config.json")]

    with patch("transformers.AutoProcessor.from_pretrained", return_value=FakeProcessor()):
        saved_filepaths = olive_model.save_metadata(tmp_path)

    # the pre-existing tokenizer config is untouched, the processor config is new
    assert (tmp_path / "tokenizer_config.json").read_text() == custom_tokenizer_config
    assert (tmp_path / "preprocessor_config.json").read_text() == '{"image_processor": true}'
    assert str(tmp_path / "preprocessor_config.json") in saved_filepaths
    assert str(tmp_path / "tokenizer_config.json") not in saved_filepaths


@pytest.mark.parametrize("trust_remote_code", [True, False])
def test_save_metadata_with_module_files(trust_remote_code, tmp_path):
    load_kwargs = {"trust_remote_code": trust_remote_code, "revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"}
    olive_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        load_kwargs=load_kwargs,
    )

    saved_filepaths = olive_model.save_metadata(tmp_path)
    assert all(Path(fp).exists() for fp in saved_filepaths)
    config = transformers.AutoConfig.from_pretrained(tmp_path, **load_kwargs)
    assert config.__class__.__name__ == "Phi3Config"
    if trust_remote_code:
        assert config.__module__.startswith(f"transformers_modules.{tmp_path.name}.")
        assert config.__module__.endswith(".configuration_phi3")
    else:
        assert config.__module__ == "transformers.models.phi3.configuration_phi3"
    assert isinstance(
        transformers.AutoTokenizer.from_pretrained(tmp_path, **load_kwargs),
        transformers.PreTrainedTokenizerBase,
    )


class TestHFDummyInput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"
        self.io_config = {
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "input_shapes": [[1, 128], [1, 128], [1, 128]],
            "input_types": ["int64", "int64", "int64"],
            "output_names": ["output"],
            "dynamic_axes": {
                "input_ids": {"0": "batch_size", "1": "seq_length"},
                "attention_mask": {"0": "batch_size", "1": "seq_length"},
                "token_type_ids": {"0": "batch_size", "1": "seq_length"},
            },
            "dynamic_shapes": {
                "input_ids": {"0": "batch_size", "1": ["seq_length", 1, 256]},
                "attention_mask": {"0": "batch_size", "1": "seq_length"},
                "token_type_ids": {"0": "batch_size", "1": "seq_length"},
            },
        }

    def test_dummy_input_with_kv_cache(self):
        io_config = self.io_config
        io_config["kv_cache"] = True
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dummy_input_with_kv_cache_dict(self):
        io_config = self.io_config
        io_config["kv_cache"] = {"batch_size": 1}
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dynamic_shapes_is_generated_when_kv_cache_is_true(self):
        io_config = self.io_config
        io_config["kv_cache"] = True
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        io_config = olive_model.io_config
        assert "dynamic_shapes" in io_config
        assert "past_key_values" in io_config["dynamic_shapes"]
        assert len(io_config["dynamic_shapes"]["past_key_values"]) == 5
        assert len(io_config["dynamic_shapes"]["past_key_values"][0]) == 2
        assert io_config["dynamic_shapes"]["past_key_values"][0][0] == {0: "batch_size", 2: "past_sequence_length"}

    def test_dict_io_config(self):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=self.io_config)
        # get io config
        io_config = olive_model.io_config
        assert io_config == IoConfig(**self.io_config).model_dump(exclude_none=True)

    @patch("olive.model.handler.mixin.hf.get_model_io_config")
    def test_hf_config_io_config(self, get_model_io_config):
        get_model_io_config.return_value = self.io_config
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        # get io config
        io_config = olive_model.io_config
        assert io_config == self.io_config
        get_model_io_config.assert_called_once_with(
            self.model_name,
            self.task,
            olive_model.load_model(),
            test_model_config=None,
        )

    @patch("olive.data.template.dummy_data_config_template")
    def test_input_shapes_dummy_inputs(self, dummy_data_config_template):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=self.io_config)

        dummy_data_config_template.return_value.to_data_container.return_value.get_first_batch.return_value = 1, 0

        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        dummy_data_config_template.assert_called_once_with(
            input_shapes=self.io_config["input_shapes"],
            input_types=self.io_config["input_types"],
            input_names=self.io_config["input_names"],
        )
        dummy_data_config_template.return_value.to_data_container.assert_called_once()
        dummy_data_config_template.return_value.to_data_container.return_value.get_first_batch.assert_called_once()
        assert dummy_inputs == 1

    @patch("olive.model.handler.mixin.hf.get_model_dummy_input")
    def test_hf_onnx_config_dummy_inputs(self, get_model_dummy_input):
        get_model_dummy_input.return_value = 1
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        get_model_dummy_input.assert_called_once_with(
            self.model_name,
            self.task,
            model=ANY,
            test_model_config=None,
        )
        assert dummy_inputs == 1
