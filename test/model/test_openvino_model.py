# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from olive.model.handler.openvino import create_openvino_core


def test_create_openvino_core_registers_group_query_attention_extension():
    core = MagicMock()
    extension = MagicMock()
    openvino = SimpleNamespace(
        Core=MagicMock(return_value=core),
        op=SimpleNamespace(_GroupQueryAttentionExtension=extension),
    )

    with patch.dict(sys.modules, {"openvino": openvino}):
        result = create_openvino_core()

    assert result is core
    extension.assert_called_once_with()
    core.add_extension.assert_called_once_with(extension.return_value)


def test_create_openvino_core_without_group_query_attention_extension():
    core = MagicMock()
    openvino = SimpleNamespace(Core=MagicMock(return_value=core), op=SimpleNamespace())

    with patch.dict(sys.modules, {"openvino": openvino}):
        result = create_openvino_core()

    assert result is core
    core.add_extension.assert_not_called()
