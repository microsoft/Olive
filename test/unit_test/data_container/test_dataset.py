# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from datasets import Dataset as HFDataset

from olive.data.component.dataset import BaseDataset, DummyDataset


def get_dict_dataset(length=256, max_samples=None):
    label_name = "original_label"
    input_names = ["input_1", "input_2"]
    data = []
    for i in range(length):
        data.append({input_names[0]: [i, i], input_names[1]: [i, i, i], label_name: i})  # noqa: PERF401
    return BaseDataset(data, "original_label", max_samples)


def get_dummy_dataset(length=256):
    input_shapes = [[2], [3]]
    input_names = ["input_1", "input_2"]
    return DummyDataset(input_shapes, input_names, max_samples=length)


def get_hf_dataset():
    length = 300
    data = {"input_1": [], "input_2": [], "original_label": []}
    for i in range(length):
        data["input_1"].append([i, i])
        data["input_2"].append([i, i, i])
        data["original_label"].append(i)
    hf_dataset = HFDataset.from_dict(data)
    hf_dataset.set_format(type="torch", output_all_columns=True)
    return BaseDataset(hf_dataset, "original_label", max_samples=256)


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
