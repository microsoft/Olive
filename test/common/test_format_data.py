# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import pytest
import torch

from olive.common.utils import format_data


def _io_config(input_names, input_types):
    return {"input_names": list(input_names), "input_types": list(input_types)}


@pytest.mark.parametrize("target_type", ["bfloat16", "float16", "float32"])
def test_format_data_returns_correct_values_when_bfloat16_input(target_type):
    # bf16 -> fp32 is lossless, so the expected values are the fp32 upcast cast to the target dtype.
    tensor = torch.tensor([[1.5, -2.25], [0.0, 3.75]], dtype=torch.bfloat16)
    io_config = _io_config(["past_key"], [target_type])

    result = format_data({"past_key": tensor}, io_config)

    expected = np.ascontiguousarray(tensor.to(torch.float32).cpu().numpy(), dtype=target_type)
    assert result["past_key"].dtype == np.dtype(target_type)
    np.testing.assert_array_equal(result["past_key"], expected)


def test_format_data_does_not_raise_when_bfloat16_input():
    # Regression: torch.Tensor.numpy() on a bfloat16 tensor previously raised
    # "TypeError: Got unsupported ScalarType BFloat16".
    tensor = torch.ones((2, 3), dtype=torch.bfloat16)
    io_config = _io_config(["past_value"], ["bfloat16"])

    result = format_data({"past_value": tensor}, io_config)

    assert result["past_value"].shape == (2, 3)


@pytest.mark.parametrize(
    ("torch_dtype", "target_type"),
    [
        (torch.float16, "float16"),
        (torch.float32, "float32"),
        (torch.int64, "int64"),
        (torch.int32, "int32"),
    ],
)
def test_format_data_is_byte_identical_when_non_bfloat16_tensor(torch_dtype, target_type):
    tensor = torch.arange(6, dtype=torch_dtype).reshape(2, 3)
    io_config = _io_config(["input"], [target_type])

    result = format_data({"input": tensor}, io_config)

    expected = np.ascontiguousarray(tensor.cpu().numpy(), dtype=target_type)
    assert result["input"].dtype == np.dtype(target_type)
    assert result["input"].tobytes() == expected.tobytes()


def test_format_data_is_byte_identical_when_numpy_array_input():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    io_config = _io_config(["input"], ["float32"])

    result = format_data({"input": array}, io_config)

    expected = np.ascontiguousarray(array, dtype="float32")
    assert result["input"].tobytes() == expected.tobytes()
