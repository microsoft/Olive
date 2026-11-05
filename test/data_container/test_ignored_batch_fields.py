# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import pickle

import pytest
import torch
from torch.utils.data import Dataset

from olive.data.component.dataloader import (
    dataloader_with_ignored_batch_fields,
    default_dataloader,
    no_auto_batch_dataloader,
)
from olive.data.config import DataConfig


class _FieldDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index, index + 1]),
            "step": 7,
            "total_seq_len": 9,
            "worker_pid": os.getpid(),
        }, index


@pytest.mark.parametrize("fields", ["step", ["step", "total_seq_len"]])
def test_ignored_field_collator_preserves_values_after_pickle_roundtrip(fields):
    loader = dataloader_with_ignored_batch_fields(_FieldDataset(), fields_no_batch=fields, batch_size=2)
    collate = pickle.loads(pickle.dumps(loader.collate_fn))
    data, labels = collate([loader.dataset[0], loader.dataset[1]])
    assert isinstance(data["step"], int)
    assert data["step"] == 7
    assert torch.equal(data["input_ids"], torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(labels, torch.tensor([0, 1]))
    if isinstance(fields, list):
        assert isinstance(data["total_seq_len"], int)
        assert data["total_seq_len"] == 9
    else:
        assert torch.equal(data["total_seq_len"], torch.tensor([9, 9]))


@pytest.mark.parametrize("fields", ["step", ["step", "total_seq_len"]])
@pytest.mark.parametrize("workers", [0, 1], ids=["main_process", "spawn_worker"])
def test_ignored_field_dataloader_preserves_batches_when_using_workers(fields, workers):
    kwargs = {"multiprocessing_context": "spawn", "timeout": 120} if workers else {}
    loader = dataloader_with_ignored_batch_fields(
        _FieldDataset(), fields_no_batch=fields, batch_size=2, num_workers=workers, **kwargs
    )
    batches = list(loader)
    assert len(batches) == 2
    for batch_index, (data, labels) in enumerate(batches):
        start = 2 * batch_index
        assert isinstance(data["step"], int)
        assert data["step"] == 7
        assert torch.equal(data["input_ids"], torch.tensor([[start, start + 1], [start + 1, start + 2]]))
        assert torch.equal(labels, torch.tensor([start, start + 1]))
        assert bool(torch.all(data["worker_pid"] != os.getpid())) is bool(workers)
        if isinstance(fields, list):
            assert isinstance(data["total_seq_len"], int)
            assert data["total_seq_len"] == 9
        else:
            assert torch.equal(data["total_seq_len"], torch.tensor([9, 9]))


def test_configured_dataloader_preserves_scalar_field_when_using_spawn_worker():
    config = DataConfig(
        name="spawn_scalar_fields",
        dataloader_config={
            "type": "dataloader_with_ignored_batch_fields",
            "params": {
                "fields_no_batch": "step",
                "batch_size": 2,
                "num_workers": 1,
                "multiprocessing_context": "spawn",
                "timeout": 120,
            },
        },
    )
    batches = list(config.to_data_container().dataloader(_FieldDataset()))
    assert len(batches) == 2
    assert all(data["step"] == 7 for data, _ in batches)
    assert all(bool(torch.all(data["worker_pid"] != os.getpid())) for data, _ in batches)


def test_default_dataloader_preserves_batch_dimension_for_scalar_fields():
    data, labels = next(iter(default_dataloader(_FieldDataset(), batch_size=2)))
    assert torch.equal(data["step"], torch.tensor([7, 7]))
    assert torch.equal(labels, torch.tensor([0, 1]))


def test_no_auto_batch_dataloader_preserves_individual_samples():
    data, label = next(iter(no_auto_batch_dataloader(_FieldDataset(), batch_size=2)))
    assert data["step"] == 7
    assert data["input_ids"].shape == (2,)
    assert label == 0
