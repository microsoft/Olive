# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_sc = logging.StreamHandler(stream=sys.stdout)
_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
_sc.setFormatter(_formatter)
_logger.addHandler(_sc)
_logger.propagate = False

__version__ = "0.9.1"

try:
    import onnxruntime as ort
    from onnxruntime import winml  # noqa: F401 # pylint: disable=unused-import

    ort._get_available_providers = ort.get_available_providers

    def get_available_providers_winml():
        # pylint: disable=protected-access
        providers = ort._get_available_providers()
        extra_providers = {ep_device.ep_name for ep_device in ort.get_ep_devices()} - set(providers)
        return providers + list(extra_providers)

    ort.get_available_providers = get_available_providers_winml
except:
    pass
