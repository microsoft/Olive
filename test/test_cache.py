# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import ANY, mock_open, patch

import pytest

from olive.cache import CacheConfig, OliveCache, SharedCache, isolated_cache_env
from olive.common.constants import DEFAULT_WORKFLOW_ID
from olive.systems.utils import create_new_environ

# pylint: disable=W0201


class TestCache:
    def test_cache_env_uses_process_environment_outside_isolated_context(self, tmp_path, monkeypatch):
        first_cache_dir = tmp_path / "first"
        second_cache_dir = tmp_path / "second"
        OliveCache({"cache_dir": first_cache_dir}).set_cache_env()
        monkeypatch.setenv("OLIVE_CACHE_DIR", str(second_cache_dir))

        assert OliveCache.from_cache_env().get_cache_dir() == second_cache_dir

    def test_cache_env_is_scoped_to_parallel_build_thread(self, tmp_path, monkeypatch):
        barrier = Barrier(2)
        process_cache_dir = tmp_path / "process"
        monkeypatch.setenv("OLIVE_CACHE_DIR", str(process_cache_dir))

        def set_and_read_cache(cache_dir):
            with isolated_cache_env(cache_dir):
                cache = OliveCache({"cache_dir": cache_dir})
                cache.set_cache_env()
                barrier.wait(timeout=2)
                return (
                    OliveCache.from_cache_env().get_cache_dir(),
                    Path(create_new_environ()["OLIVE_CACHE_DIR"]),
                    Path(os.environ["OLIVE_CACHE_DIR"]),
                )

        cache_dirs = [tmp_path / "first", tmp_path / "second"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            resolved = list(executor.map(set_and_read_cache, cache_dirs))

        assert [item[0] for item in resolved] == [path.resolve() for path in cache_dirs]
        assert [item[1] for item in resolved] == cache_dirs
        assert all(item[2] == process_cache_dir for item in resolved)
        assert Path(os.environ["OLIVE_CACHE_DIR"]) == process_cache_dir

    @pytest.mark.parametrize("clean_cache", [True, False])
    def test_cache_init(self, clean_cache, tmp_path):
        # setup
        cache_dir = tmp_path / "cache_dir"
        runs_dir = cache_dir / DEFAULT_WORKFLOW_ID / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        dummy_file = runs_dir / "0_p(･◡･)p.json"
        with open(dummy_file, "w") as _:
            pass

        # execute
        CacheConfig(cache_dir=cache_dir, clean_cache=clean_cache).create_cache()

        # assert
        assert cache_dir.exists()
        assert runs_dir.exists()
        assert clean_cache == (not dummy_file.exists())

    @pytest.mark.parametrize(
        ("pass_name", "pass_config", "input_model_id", "accelerator_spec", "expected"),
        [
            (
                "pass_name",
                {"param": "value"},
                "model",
                "GPU",
                {
                    "input_model_id": "model",
                    "pass_name": "pass_name",
                    "pass_config": {"param": "value"},
                    "accelerator_spec": "GPU",
                },
            ),
            (
                "pass_name",
                {},
                "model",
                None,
                {"input_model_id": "model", "pass_name": "pass_name", "pass_config": {}, "accelerator_spec": None},
            ),
        ],
    )
    def test_get_run_json(self, pass_name, pass_config, input_model_id, accelerator_spec, expected):
        # execute
        result = OliveCache.get_run_json(pass_name, pass_config, input_model_id, accelerator_spec)

        # assert
        assert result == expected

    def test_cache_model(self, tmp_path):
        # setup
        model_id = "model"
        model_json = {"config": {"model_path": "path/to/model"}}
        cache = CacheConfig(cache_dir=tmp_path).create_cache()

        # execute
        cache.cache_model(model_id, model_json)

        # assert
        model_json_path = cache.get_model_json_path(model_id)
        assert model_json_path.exists()
        with model_json_path.open() as f:
            cached_model_json = json.load(f)
        assert cached_model_json == model_json

    @patch("olive.cache.SharedCache.cache_model")
    def test_cache_model_with_shared_cache(self, mock_shared_cache_model, tmp_path):
        # setup
        model_id = "model_1"
        model_json = {"config": {"model_path": "path/to/model_1"}}
        cache_config = CacheConfig(cache_dir=tmp_path, enable_shared_cache=True)
        cache = cache_config.create_cache()

        # execute
        cache.cache_model(model_id, model_json)

        # assert
        model_json_path = cache.get_model_json_path(model_id)
        assert model_json_path.exists()
        with model_json_path.open() as f:
            cached_model_json = json.load(f)
        assert cached_model_json == model_json
        mock_shared_cache_model.assert_called_once_with(model_id, model_json)

    def test_load_model(self, tmp_path):
        # setup
        cache_config = CacheConfig(cache_dir=tmp_path)
        cache = cache_config.create_cache()
        model_id = "model"
        model_json = {"config": {"model_path": "path/to/model"}}
        model_json_path = cache.get_model_json_path(model_id)
        model_json_path.parent.mkdir(parents=True, exist_ok=True)
        with model_json_path.open("w") as f:
            json.dump(model_json, f)

        # execute
        loaded_model = cache.load_model(model_id)

        # assert
        assert loaded_model == model_json

    @patch("olive.cache.SharedCache.load_model")
    def test_load_model_with_shared_cache(self, mock_shared_cache_load_model, tmp_path):
        # setup
        model_id = "model_1"
        model_json = {"config": {"model_path": "path/to/model_1"}}
        cache_config = CacheConfig(cache_dir=tmp_path, enable_shared_cache=True)
        cache = cache_config.create_cache()
        mock_shared_cache_load_model.return_value = model_json

        # execute
        loaded_model = cache.load_model(model_id)

        # assert
        assert loaded_model == model_json
        mock_shared_cache_load_model.assert_called_once_with(model_id, cache.get_model_json_path(model_id))

    @pytest.mark.parametrize(
        ("pass_name", "pass_config", "input_model_id", "output_model_id", "accelerator_spec"),
        [
            ("pass_name", {"param": "value"}, "input_model", "output_model", "GPU"),
            ("pass_name", {}, "input_model", "output_model", None),
        ],
    )
    def test_cache_run(self, pass_name, pass_config, input_model_id, output_model_id, accelerator_spec, tmp_path):
        # setup
        cache = CacheConfig(cache_dir=tmp_path).create_cache()
        run_json_path = cache.get_run_json_path(output_model_id)

        # execute
        cache.cache_run(pass_name, pass_config, input_model_id, output_model_id, accelerator_spec)

        # assert
        assert run_json_path.exists()
        with run_json_path.open() as f:
            cached_run_json = json.load(f)
        expected_run_json = OliveCache.get_run_json(pass_name, pass_config, input_model_id, accelerator_spec)
        expected_run_json["output_model_id"] = output_model_id
        assert cached_run_json == expected_run_json

    @patch("olive.cache.SharedCache.cache_run")
    @pytest.mark.parametrize(
        ("pass_name", "pass_config", "input_model_id", "output_model_id", "accelerator_spec"),
        [
            ("pass_name", {"param": "value"}, "input_model", "output_model", "GPU"),
            ("pass_name", {}, "input_model", "output_model", None),
        ],
    )
    def test_cache_run_with_shared_cache(
        self, mock_shared_cache_run, pass_name, pass_config, input_model_id, output_model_id, accelerator_spec, tmp_path
    ):
        # setup
        cache_config = CacheConfig(cache_dir=tmp_path, enable_shared_cache=True)
        cache = cache_config.create_cache()
        run_json_path = cache.get_run_json_path(output_model_id)

        # execute
        cache.cache_run(pass_name, pass_config, input_model_id, output_model_id, accelerator_spec)

        # assert
        assert run_json_path.exists()
        with run_json_path.open() as f:
            cached_run_json = json.load(f)
        expected_run_json = OliveCache.get_run_json(pass_name, pass_config, input_model_id, accelerator_spec)
        expected_run_json["output_model_id"] = output_model_id
        assert cached_run_json == expected_run_json
        mock_shared_cache_run.assert_called_once_with(output_model_id, run_json_path)

    def test_load_run_from_model_id(self, tmp_path):
        # setup
        cache_config = CacheConfig(cache_dir=tmp_path)
        cache = cache_config.create_cache()
        model_id = "model"
        run_json = {"key": "value"}
        run_json_path = cache.get_run_json_path(model_id)
        run_json_path.parent.mkdir(parents=True, exist_ok=True)
        with run_json_path.open("w") as f:
            json.dump(run_json, f)

        # execute
        loaded_run = cache.load_run_from_model_id(model_id)

        # assert
        assert loaded_run == run_json

    @patch("olive.cache.SharedCache.load_run")
    def test_load_run_from_model_id_with_shared_cache(self, mock_shared_cache_load_run, tmp_path):
        # setup
        model_id = "model_1"
        run_json = {"key": "value"}
        cache_config = CacheConfig(cache_dir=tmp_path, enable_shared_cache=True)
        cache = cache_config.create_cache()
        mock_shared_cache_load_run.return_value = run_json

        # execute
        loaded_run = cache.load_run_from_model_id(model_id)

        # assert
        assert loaded_run == run_json
        mock_shared_cache_load_run.assert_called_once_with(model_id, cache.get_run_json_path(model_id))

    def test_load_run_from_model_id_not_found(self, tmp_path):
        # setup
        cache_config = CacheConfig(cache_dir=tmp_path)
        cache = cache_config.create_cache()
        model_id = "non_existent_model"

        # execute
        loaded_run = cache.load_run_from_model_id(model_id)

        # assert
        assert loaded_run == {}

    def test_load_run_from_model_id_invalid_json(self, tmp_path):
        # setup
        cache_config = CacheConfig(cache_dir=tmp_path)
        cache = cache_config.create_cache()
        model_id = "model"
        run_json_path = cache.get_run_json_path(model_id)
        run_json_path.parent.mkdir(parents=True, exist_ok=True)
        with run_json_path.open("w") as f:
            f.write("invalid json")

        # execute
        loaded_run = cache.load_run_from_model_id(model_id)

        # assert
        assert loaded_run == {}

    def test_cache_evaluation(self, tmp_path):
        # setup
        model_id = "model"
        evaluation_json = {"accuracy": 0.95}
        cache = CacheConfig(cache_dir=tmp_path).create_cache()

        # execute
        cache.cache_evaluation(model_id, evaluation_json)

        # assert
        evaluation_json_path = cache.get_evaluation_json_path(model_id)
        assert evaluation_json_path.exists()
        with evaluation_json_path.open() as f:
            cached_evaluation_json = json.load(f)
        assert cached_evaluation_json == evaluation_json

    @pytest.mark.parametrize(
        "model_path",
        ["model_folder", "model.onnx"],
    )
    def test_save_model(self, model_path, tmp_path):
        # setup
        model_id = "model"
        cache = CacheConfig(cache_dir=tmp_path / "cache").create_cache()

        model_parent = tmp_path / "model_parent"
        model_parent.mkdir(parents=True, exist_ok=True)

        if model_path == "model_folder":
            model_folder = model_parent / model_path
            model_folder.mkdir(parents=True, exist_ok=True)
            # create a model .onnx file and external data file
            onnx_file = model_folder / "model.onnx"
            onnx_file.touch()
            external_data_file = model_folder / "model.onnx.data"
            external_data_file.touch()
            model_p = str(model_folder)
            model_json = {"type": "onnxmodel", "config": {"model_path": model_p, "onnx_file_name": "model.onnx"}}
        else:
            model_p = str(model_parent / model_path)
            Path(model_p).touch()
            model_json = {"type": "onnxmodel", "config": {"model_path": model_p}}

        # cache model to cache_dir
        model_cache_file_path = cache.get_model_json_path(model_id)
        with open(model_cache_file_path, "w") as f:
            json.dump(model_json, f)

        # output model to output_dir
        output_dir = tmp_path / "output"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_json = cache.save_model(model_id, output_dir, True)

        # Expected output depends on whether model has external data
        if model_path == "model_folder":
            # Model has external data → folder + onnx_file_name
            expected_output_path = str(output_dir.resolve())
            assert output_json["config"]["model_path"] == expected_output_path
            assert output_json["config"]["onnx_file_name"] == "model.onnx"
            assert (output_dir / "model.onnx").exists()
            assert (output_dir / "model.onnx.data").exists()
        else:
            # Model has no external data → single file path
            expected_output_path = output_dir / "model.onnx"
            expected_output_path = str(expected_output_path.resolve())
            assert output_json["config"]["model_path"] == expected_output_path
            assert "onnx_file_name" not in output_json["config"]
            assert Path(expected_output_path).exists()

        output_json_path = output_dir / "model_config.json"
        assert output_json_path.exists()

    def test_save_model_with_custom_onnx_filename(self, tmp_path):
        # setup
        model_id = "model"
        cache = CacheConfig(cache_dir=tmp_path / "cache").create_cache()

        model_parent = tmp_path / "model_parent"
        model_parent.mkdir(parents=True, exist_ok=True)

        model_p = str(model_parent / "source.onnx")
        Path(model_p).touch()

        model_cache_file_path = cache.get_model_json_path(model_id)
        model_json = {"type": "onnxmodel", "config": {"model_path": model_p}}
        with open(model_cache_file_path, "w") as f:
            json.dump(model_json, f)

        output_models_dir = tmp_path / "output"
        output_models_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_models_dir / "custom.onnx"
        output_json = cache.save_model(model_id, output_file_path, True)

        expected_output_path = str(output_file_path.resolve())

        # assert
        assert output_json["config"]["model_path"] == expected_output_path
        assert output_file_path.exists()

        output_json_path = output_models_dir / "model_config.json"
        assert output_json_path.exists()
        with open(output_json_path) as f:
            assert expected_output_path == json.load(f)["config"]["model_path"]

    def test_save_model_no_flatten_rebases_component_and_additional_file_paths(self, tmp_path):
        # setup
        model_id = "composite_model"
        cache = CacheConfig(cache_dir=tmp_path / "cache").create_cache()

        source_dir = tmp_path / "source_model"
        decoder_dir = source_dir / "decoder"
        embedding_dir = source_dir / "embedding"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        (decoder_dir / "model.onnx").write_text("decoder")
        (embedding_dir / "model.onnx").write_text("embedding")
        (decoder_dir / "decoder_local.txt").write_text("decoder local")
        (source_dir / "genai_config.json").write_text("{}")
        (source_dir / "tokenizer.json").write_text("{}")

        model_json = {
            "type": "compositemodel",
            "config": {
                "model_path": str(source_dir),
                "model_component_names": ["decoder", "embedding"],
                "model_components": [
                    {
                        "type": "onnxmodel",
                        "config": {
                            "model_path": str(decoder_dir),
                            "onnx_file_name": "model.onnx",
                            "model_attributes": {"additional_files": [str(decoder_dir / "decoder_local.txt")]},
                        },
                    },
                    {
                        "type": "onnxmodel",
                        "config": {"model_path": str(embedding_dir), "onnx_file_name": "model.onnx"},
                    },
                ],
                "model_attributes": {
                    "no_flatten": True,
                    "additional_files": [str(source_dir / "genai_config.json"), str(source_dir / "tokenizer.json")],
                },
            },
        }
        with cache.get_model_json_path(model_id).open("w") as f:
            json.dump(model_json, f)

        output_dir = tmp_path / "output"
        output_json = cache.save_model(model_id, output_dir, True)

        # assert copied layout
        assert (output_dir / "decoder" / "model.onnx").exists()
        assert (output_dir / "embedding" / "model.onnx").exists()
        assert (output_dir / "genai_config.json").exists()
        assert (output_dir / "tokenizer.json").exists()

        # assert rewritten config paths
        assert output_json["config"]["model_path"] == str(output_dir)
        decoder_component = output_json["config"]["model_components"][0]["config"]
        assert decoder_component["model_path"] == str(output_dir / "decoder")
        assert decoder_component["onnx_file_name"] == "model.onnx"
        assert decoder_component["model_attributes"]["additional_files"] == [
            str(output_dir / "decoder" / "decoder_local.txt")
        ]

        assert output_json["config"]["model_attributes"]["additional_files"] == [
            str(output_dir / "genai_config.json"),
            str(output_dir / "tokenizer.json"),
        ]


class TestSharedCache:
    @pytest.fixture(autouse=True)
    @patch("olive.common.container_client_factory.AzureContainerClientFactory")
    def setup(self, mock_AzureContainerClientFactory):
        self.mock_factory_instance = mock_AzureContainerClientFactory.return_value
        self.shared_cache = SharedCache("dummy_account", "dummy_container")

    @patch("olive.cache.AzureContainerClientFactory.upload_blob")
    def test_cache_run(self, mock_upload_blob, tmp_path):
        # setup
        model_id = "model"
        run_json_path = tmp_path / "run.json"
        run_json_path.write_text(json.dumps({"key": "value"}))

        # execute
        self.shared_cache.cache_run(model_id, run_json_path)

        # assert
        self.shared_cache.container_client_factory.upload_blob.assert_called_once_with(f"{model_id}/run.json", ANY)

    @patch("olive.cache.AzureContainerClientFactory.upload_blob")
    @patch("olive.cache.AzureContainerClientFactory.delete_blob")
    def test_cache_run_exception(self, mock_delete_blob, mock_upload_blob, tmp_path):
        # setup
        model_id = "model"
        run_json_path = tmp_path / "run.json"
        run_json_path.write_text(json.dumps({"key": "value"}))

        # execute
        with patch.object(self.shared_cache.container_client_factory, "upload_blob", side_effect=Exception):
            self.shared_cache.cache_run(model_id, run_json_path)

            # assert
            self.shared_cache.container_client_factory.delete_blob.assert_called_once_with(model_id)

    @patch("olive.cache.AzureContainerClientFactory.exists")
    def test_load_run_not_found(self, mock_exists, tmp_path):
        # setup
        model_id = "model"
        run_json_path = tmp_path / "run.json"
        mock_exists.return_value = False

        # execute
        loaded_run = self.shared_cache.load_run(model_id, run_json_path)

        # assert
        assert loaded_run == {}

    @patch("olive.cache.AzureContainerClientFactory.exists")
    @patch("olive.cache.AzureContainerClientFactory.download_blob")
    def test_load_run(self, mock_download_blob, mock_exists, tmp_path):
        # setup
        model_id = "model"
        run_json_path = tmp_path / "run.json"
        run_json = {"key": "value"}
        mock_exists.return_value = True
        run_json_path.write_text(json.dumps(run_json))

        # execute
        loaded_run = self.shared_cache.load_run(model_id, run_json_path)

        # assert
        assert loaded_run == run_json
        mock_download_blob.assert_called_once_with(f"{model_id}/run.json", run_json_path)

    @patch("olive.cache.AzureContainerClientFactory.exists")
    @patch("olive.cache.AzureContainerClientFactory.upload_blob")
    def test_cache_model(self, mock_upload_blob, mock_exists):
        # setup
        model_id = "model_id"
        model_json = {"type": "onnxmodel", "config": {"model_path": "model.onnx"}}
        model_binary_data = b"Test binary data"
        m = mock_open(read_data=model_binary_data)
        mock_exists.return_value = False
        with patch("builtins.open", m), patch.object(Path, "exists", return_value=True):
            with open("model.onnx", "rb") as model_data:
                model_data.read()

            # execute
            self.shared_cache.cache_model(model_id, model_json)

            # assert
            # 1. model_config.json 2. model
            assert mock_upload_blob.call_count == 2

    @patch("olive.cache.AzureContainerClientFactory.exists")
    @patch("olive.cache.AzureContainerClientFactory.upload_blob")
    @patch("olive.cache.AzureContainerClientFactory.delete_blob")
    def test_cache_model_exception(self, mock_delete_blob, mock_upload_blob, mock_exists):
        # setup
        model_id = "model_id"
        model_json = {"type": "onnxmodel", "config": {"model_path": "model.onnx"}}
        model_binary_data = b"Test binary data"
        m = mock_open(read_data=model_binary_data)
        mock_exists.return_value = False
        with patch("builtins.open", m), patch.object(Path, "exists", return_value=True):
            with open("model.onnx", "rb") as model_data:
                model_data.read()

            # execute
            with patch.object(self.shared_cache.container_client_factory, "upload_blob", side_effect=Exception):
                self.shared_cache.cache_model(model_id, model_json)

            # assert
            self.shared_cache.container_client_factory.delete_blob.assert_called_once_with(model_id)

    @patch("olive.cache.AzureContainerClientFactory.exists")
    @patch("olive.cache.AzureContainerClientFactory.download_blob")
    def test_load_model(self, mock_download_blob, mock_exists, tmp_path):
        # setup
        model_id = "model"
        model_json_path = tmp_path / "model.json"
        model_json = {"config": {"model_path": "path/to/model"}}
        model_json_path.write_text(json.dumps(model_json))
        mock_exists.return_value = True

        # execute
        loaded_model = self.shared_cache.load_model(model_id, model_json_path)

        # assert
        assert loaded_model == model_json
        mock_download_blob.assert_called_once()

    @pytest.mark.parametrize("expected_exists", [True, False])
    @patch("olive.cache.AzureContainerClientFactory.exists")
    def test_exist_in_shared_cache(self, mock_exists, expected_exists):
        # setup
        blob_name = "model_id/model.json"
        mock_exists.return_value = expected_exists

        # execute
        actual_exists = self.shared_cache.exist_in_shared_cache(blob_name)

        # assert
        assert actual_exists == expected_exists
        mock_exists.assert_called_once_with(blob_name)

    @patch("olive.cache.AzureContainerClientFactory.upload_blob")
    def test_upload_model_files(self, mock_upload_blob, tmp_path):
        # setup
        model_path = tmp_path / "model"
        model_path.mkdir()
        file_path = model_path / "file.txt"
        file_path.write_text("dummy content")
        model_blob = "model_blob"

        # execute
        self.shared_cache.upload_model_files(str(model_path), model_blob)

        # assert
        mock_upload_blob.assert_called_once_with(f"{model_blob}/file.txt", ANY)
