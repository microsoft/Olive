# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect

import pytest

from olive.passes.pytorch.train_utils import get_calibration_data_config, get_calibration_dataset


@pytest.mark.parametrize("func", [get_calibration_dataset, get_calibration_data_config])
def test_calibration_helpers_default_split_is_full_train(func):
    # execute
    default_split = inspect.signature(func).parameters["split"].default

    # assert
    # measured with the gpt2 tokenizer: wikitext-2-raw-v1 'train[:1000]' is only 61,816 tokens (353 of the
    # first 1000 rows are blank lines or headers) which is ~30 blocks of 2048 tokens, silently
    # under-delivering the requested max_samples. The full 'train' split gives ~1,167 blocks.
    assert default_split == "train"


def test_get_calibration_data_config_uses_default_split():
    # execute
    data_config = get_calibration_data_config("dummy-model")

    # assert
    assert data_config.load_dataset_config.params["split"] == "train"
