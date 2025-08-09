# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import pytest

from olive.common.utils import WeightsFileFormat, load_weights, save_weights


def test_invalid_file_format():
    with pytest.raises(ValueError, match="is not a valid WeightsFileFormat"):
        save_weights({}, "dummy", "dummy")


@pytest.mark.parametrize(
    ("file_format", "expected_suffix"),
    [
        (WeightsFileFormat.NUMPY, ".npz"),
        (WeightsFileFormat.PT, ".pt"),
        (WeightsFileFormat.SAFETENSORS, ".safetensors"),
    ],
)
@pytest.mark.parametrize("framework", ["numpy", "pt"])
def test_save_and_load_weights(file_format, expected_suffix, framework, tmp_path):
    weights = {}
    for dtype in [np.float32, np.float16, np.int32, np.int64, np.uint8]:
        weights[f"key_{dtype}"] = np.random.rand(2, 3).astype(dtype)

    output_path = tmp_path / "weights"
    weights_file = save_weights(weights, output_path, file_format)

    assert weights_file.exists()
    assert weights_file.suffix == expected_suffix

    loaded_weights = load_weights(weights_file, framework=framework)
    assert set(weights.keys()) == set(loaded_weights.keys())
    for key, value in weights.items():
        np.testing.assert_array_equal(
            value, loaded_weights[key] if framework == "numpy" else loaded_weights[key].numpy()
        )
