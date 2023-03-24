# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
import shutil
from pathlib import Path

import pytest

from olive.cache import clean_pass_run_cache


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

        if model_path == "0_model_folder":
            model_folder = cache_dir / model_path
            model_folder.mkdir(parents=True, exist_ok=True)
            model_p = str(model_folder)
        else:
            model_p = str(cache_dir / model_path)
            open(str(cache_dir / model_path), "w")

        run_cache_dir = cache_dir / "runs"
        run_cache_dir.mkdir(parents=True, exist_ok=True)
        run_cache_file_path = str((run_cache_dir / f"{pass_type}-p(･◡･)p.json").resolve())
        with open(run_cache_file_path, "w") as run_cache_file:
            run_data = (
                '{"pass_name": "OnnxConversion", "input_model_id": "0", "output_model_id": "0_OnnxConversion-0-1"}'
            )
            run_cache_file.write(run_data)

        model_cache_dir = cache_dir / "models"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        model_cache_file_path = str((model_cache_dir / "0_p(･◡･)p.json").resolve())
        with open(model_cache_file_path, "w") as model_cache_file:
            model_data = f'{{"model_path": "{model_p}"}}'
            if platform.system() == "Windows":
                model_data = model_data.replace("\\", "//")
            model_cache_file.write(model_data)

        evaluation_cache_dir = cache_dir / "evaluations"
        evaluation_cache_dir.mkdir(parents=True, exist_ok=True)
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
