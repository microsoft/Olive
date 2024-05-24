# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Union

from datasets import load_dataset

from olive.data.registry import Registry


@Registry.register_dataset()
def load_tiny_code_dataset(data_name: str, split: str, language: str, token: Union[bool, str] = True):
    dataset = load_dataset(data_name, split=split, token=token)
    return dataset.filter(lambda x: x["programming_language"] == language)
