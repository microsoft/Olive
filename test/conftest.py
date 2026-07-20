# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from unittest.mock import patch

import pytest
from packaging import version

from olive.telemetry.telemetry import Telemetry
from test.utils import create_onnx_model_file, delete_onnx_model_files


@pytest.fixture(scope="session", autouse=True)
def setup_onnx_model(request, tmp_path_factory):
    cache_path = tmp_path_factory.mktemp("transformers_cache")
    import transformers

    # we cannot use os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    # because the TRANSFORMERS_CACHE is loaded when importing transformers
    transformers.utils.hub.TRANSFORMERS_CACHE = str(cache_path)

    from datasets import disable_caching

    disable_caching()
    create_onnx_model_file()
    yield
    delete_onnx_model_files()
    shutil.rmtree(cache_path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def maybe_patch_inc():
    import peft

    if version.parse(peft.__version__) >= version.parse("0.16.0"):
        # peft 0.16.0+ has a new dispatcher for inc which imports missing dependencies
        with patch("peft.tuners.lora.inc.is_inc_available", new=lambda: False):
            yield
    else:
        yield


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry(tmp_path_factory):
    # Keep telemetry fully inert during tests. The device-id heartbeat is now
    # durable, so simply constructing Telemetry() would enqueue one to the real
    # store and the uploader would try to send it. Redirect the store to a
    # throwaway directory and stub the HTTP transport so no test run writes to
    # the real telemetry store or reaches the network.
    import olive.telemetry.deviceid._store as deviceid_store_module
    import olive.telemetry.library.transport as transport_module
    import olive.telemetry.telemetry as telemetry_module
    import olive.telemetry.utils as telemetry_utils

    telemetry_dir = tmp_path_factory.mktemp("telemetry")
    with (
        patch.object(telemetry_module, "get_telemetry_base_dir", lambda: telemetry_dir),
        patch.object(telemetry_utils, "get_telemetry_base_dir", lambda: telemetry_dir),
        patch.object(deviceid_store_module, "get_telemetry_base_dir", lambda: telemetry_dir),
        patch.object(transport_module.HttpJsonPostTransport, "send", lambda *args, **kwargs: (True, 204)),
    ):
        telemetry = Telemetry()
        telemetry.disable_telemetry()
        try:
            yield
        finally:
            telemetry.shutdown()
