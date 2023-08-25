# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest
from datasets import Dataset as HFDataset

from olive.data.component.dataset import BaseDataset, DummyDataset


def get_dict_dataset(length=256, max_samples=None):
    label_name = "original_label"
    input_names = ["input_1", "input_2"]
    data = []
    for i in range(length):
        data.append({input_names[0]: [i, i], input_names[1]: [i, i, i], label_name: i})
    return BaseDataset(data, ["original_label"], max_samples)


def get_dummy_dataset():
    input_shapes = [[2], [3]]
    input_names = ["input_1", "input_2"]
    return DummyDataset(input_shapes, input_names)


def get_hf_dataset():
    length = 300
    data = {"input_1": [], "input_2": [], "original_label": []}
    for i in range(length):
        data["input_1"].append([i, i])
        data["input_2"].append([i, i, i])
        data["original_label"].append(i)
    hf_dataset = HFDataset.from_dict(data)
    hf_dataset.set_format(type="torch", output_all_columns=True)
    return BaseDataset(hf_dataset, ["original_label"], max_samples=256)


class TestDataset:
    def test_base_dataset(self):
        # default
        dataset = get_dict_dataset()
        assert len(dataset) == 256
        assert dataset[0] == ({"input_1": [0, 0], "input_2": [0, 0, 0]}, 0)

        # max_samples < length
        dataset = get_dict_dataset(max_samples=200)
        assert len(dataset) == 200

        # max_samples > length
        dataset = get_dict_dataset(max_samples=300)
        assert len(dataset) == 256

    @pytest.mark.parametrize("dataset_func", [get_dict_dataset, get_dummy_dataset, get_hf_dataset])
    @pytest.mark.parametrize("label_name", [None, "original_label", "labels"])
    def test_dataset_to_hf_dataset(self, dataset_func, label_name):
        dataset = dataset_func()
        if label_name is None:
            hf_dataset = dataset.to_hf_dataset()
        else:
            hf_dataset = dataset.to_hf_dataset(label_name=label_name)
        # assert the dataset is converted to HFDataset
        assert isinstance(hf_dataset, HFDataset)
        # assert length
        assert len(hf_dataset) == 256
        # ensure all input and label names are in the features
        label_name = label_name or "label"
        for key in ["input_1", "input_2", label_name]:
            assert key in hf_dataset.features
        # assert shape of the first sample
        assert hf_dataset["input_1"][0].shape == (2,)
        assert hf_dataset["input_2"][0].shape == (3,)
        assert hf_dataset[label_name][0].shape == ()
