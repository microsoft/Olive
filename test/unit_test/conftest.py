# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import create_onnx_model_file, delete_onnx_model_files

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_onnx_model():
    from datasets import disable_caching

    disable_caching()
    create_onnx_model_file()
    yield
    delete_onnx_model_files()
