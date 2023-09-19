# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Union

from datasets import load_dataset

from olive.data.registry import Registry


# TODO: remove custom dataset component once default dataset component supports filter, tokens and split
@Registry.register_dataset()
def load_tiny_code_dataset(dataset_name: str, split: str, language: str, token: Union[bool, str] = True):
    dataset = load_dataset(dataset_name, split=split, token=token)
    dataset = dataset.filter(lambda x: x["programming_language"] == language)
    return dataset
