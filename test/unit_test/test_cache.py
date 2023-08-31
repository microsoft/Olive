# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.cache import clean_pass_run_cache, create_cache, download_resource, get_cache_sub_dirs, save_model
from olive.resource_path import AzureMLModel


class TestCache:
    @pytest.mark.parametrize(
        "model_path",
        ["0_model_folder", "0_model.onnx"],
    )
    def test_clean_pass_run_cache(self, model_path):
        # setup
        pass_type = "onnxconversion"
        cache_dir = Path("cache_dir")
        cache_dir.mkdir(parents=True, exist_ok=True)
        create_cache(cache_dir)
        model_cache_dir, run_cache_dir, evaluation_cache_dir, _ = get_cache_sub_dirs(cache_dir)

        if model_path == "0_model_folder":
            model_folder = model_cache_dir / model_path
            model_folder.mkdir(parents=True, exist_ok=True)
            model_p = str(model_folder)
        else:
            model_p = str(model_cache_dir / model_path)
            open(model_p, "w")

        run_cache_file_path = str((run_cache_dir / f"{pass_type}-p(･◡･)p.json").resolve())
        with open(run_cache_file_path, "w") as run_cache_file:
            run_data = (
                '{"pass_name": "OnnxConversion", "input_model_id": "0", "output_model_id": "0_OnnxConversion-0-1"}'
            )
            run_cache_file.write(run_data)

        model_cache_file_path = str((model_cache_dir / "0_p(･◡･)p.json").resolve())
        with open(model_cache_file_path, "w") as model_cache_file:
            model_data = f'{{"model_path": "{model_p}"}}'
            if platform.system() == "Windows":
                model_data = model_data.replace("\\", "//")
            model_cache_file.write(model_data)

        evaluation_cache_file_path = str((evaluation_cache_dir / "0_p(･◡･)p.json").resolve())
        open(evaluation_cache_file_path, "w")

        # execute
        clean_pass_run_cache(pass_type, cache_dir)

        # assert
        assert not os.path.exists(model_p)
        assert not os.path.exists(run_cache_file_path)
        assert not os.path.exists(model_cache_file_path)
        assert not os.path.exists(evaluation_cache_file_path)

        # cleanup
        shutil.rmtree(cache_dir)

    @pytest.mark.parametrize(
        "model_path",
        ["0_model_folder", "0_model.onnx"],
    )
    def test_save_model(self, model_path):
        # setup
        cache_dir = Path("cache_dir")
        cache_dir.mkdir(parents=True, exist_ok=True)
        create_cache(cache_dir)
        model_cache_dir = get_cache_sub_dirs(cache_dir)[0]

        if model_path == "0_model_folder":
            model_folder = model_cache_dir / model_path
            model_folder.mkdir(parents=True, exist_ok=True)
            model_p = str(model_folder)
            # create a dummy .onnx file in the folder
            onnx_file = model_folder / "dummy.onnx"
            onnx_file.touch()
        else:
            model_p = str(model_cache_dir / model_path)
            Path(model_p).touch()

        # cache model to cache_dir
        model_id = "0"
        model_cache_dir = cache_dir / "models"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        model_cache_file_path = str((model_cache_dir / f"{model_id}_p(･◡･)p.json").resolve())
        model_json = {"type": "onnxmodel", "config": {"model_path": model_p}}
        json.dump(model_json, open(model_cache_file_path, "w"))

        # output model to output_dir
        output_dir = cache_dir / "output"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_name = "test_model"
        output_json = save_model(model_id, output_dir, output_name, True, cache_dir)

        # assert
        output_model_path = (output_dir / f"{output_name}").with_suffix(Path(model_p).suffix).resolve()
        output_model_path = str(output_model_path.resolve())
        output_json_path = output_dir / f"{output_name}.json"
        assert output_model_path == output_json["config"]["model_path"]
        assert os.path.exists(output_model_path)
        assert os.path.exists(output_json_path)
        assert output_model_path == json.load(open(output_json_path))["config"]["model_path"]

        # cleanup
        shutil.rmtree(cache_dir)

    @patch("olive.resource_path.AzureMLModel.save_to_dir")
    def test_download_resource(self, mock_save_to_dir):
        # setup
        tmp_dir = tempfile.TemporaryDirectory()
        cache_dir = Path(tmp_dir.name) / "cache_dir"
        cache_dir2 = Path(tmp_dir.name) / "cache_dir2"

        # resource_path
        resource_path = AzureMLModel(
            {
                "azureml_client": {
                    "workspace_name": "dummy_workspace_name",
                    "subscription_id": "dummy_subscription_id",
                    "resource_group": "dummy_resource_group",
                },
                "name": "dummy_model_name",
                "version": "dummy_model_version",
            }
        )

        mock_save_to_dir.return_value = "dummy_string_name"

        # execute
        # first time
        cached_path = download_resource(resource_path, cache_dir)
        assert cached_path.get_path() == "dummy_string_name"
        assert mock_save_to_dir.call_count == 1

        # second time
        cached_path = download_resource(resource_path, cache_dir)
        assert cached_path.get_path() == "dummy_string_name"
        # uses cached value so save_to_dir is not called again
        assert mock_save_to_dir.call_count == 1

        # change cache_dir
        cached_path = download_resource(resource_path, cache_dir2)
        assert cached_path.get_path() == "dummy_string_name"
        assert mock_save_to_dir.call_count == 2
