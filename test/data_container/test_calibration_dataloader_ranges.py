# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import onnx
import pytest
import torch
from torch.utils.data import DataLoader

from olive.data.component.dataloader import default_calibration_dataloader


def make_reader(with_labels=False):
    samples = [{"input": torch.tensor([float(index), float(index)])} for index in range(6)]
    dataset = [(sample, index) for index, sample in enumerate(samples)] if with_labels else samples
    return default_calibration_dataloader(DataLoader(dataset, batch_size=1, shuffle=False))


def read_values(reader):
    return [batch["input"][0, 0].item() for batch in reader]


@pytest.mark.parametrize(("start", "end"), [(0, 3), (2, 4), (4, 6), (2, 2), (5, 9)])
@pytest.mark.parametrize("with_labels", [False, True])
def test_set_range_selects_requested_batches(start, end, with_labels):
    reader = make_reader(with_labels)
    reader.set_range(start, end)
    assert read_values(reader) == list(range(6))[start:end]
    assert reader.get_next() is None
    assert len(reader) == 6


@pytest.mark.parametrize("ranges", [[(3, 5), (1, 3)], [(0, 2), (4, 6)], [(2, 4), (2, 4)], [(0, 2), (2, 4), (4, 6)]])
def test_set_range_repositions_after_previous_window(ranges):
    reader = make_reader()
    for start, end in ranges:
        reader.set_range(start, end)
        assert read_values(reader) == list(range(start, end))


@pytest.mark.parametrize("consumed", [0, 1, 3])
def test_rewind_replays_batches_within_existing_end_bound(consumed):
    reader = make_reader()
    reader.set_range(0, 3)
    for _ in range(consumed):
        assert reader.get_next() is not None
    reader.rewind()
    assert read_values(reader) == [0.0, 1.0, 2.0]
    reader.rewind()
    assert read_values(reader) == [0.0, 1.0, 2.0]


def test_rewind_returns_to_dataset_start_after_nonzero_range():
    reader = make_reader()
    reader.set_range(2, 4)
    assert read_values(reader) == [2.0, 3.0]
    reader.rewind()
    assert read_values(reader) == [0.0, 1.0, 2.0, 3.0]


def test_unbounded_reader_preserves_complete_iteration():
    reader = make_reader()
    assert read_values(reader) == list(range(6))
    reader.rewind()
    assert read_values(reader) == list(range(6))


def test_selected_range_controls_actual_calibration_extrema(tmp_path):
    from onnxruntime.quantization.calibrate import MinMaxCalibrater

    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "calibration_window",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=9)
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, model_path)
    reader = make_reader()
    reader.set_range(2, 4)
    calibrater = MinMaxCalibrater(
        model_path, op_types_to_calibrate=["MatMul"], augmented_model_path=str(tmp_path / "augmented.onnx")
    )
    calibrater.augment_graph()
    calibrater.create_inference_session()
    calibrater.collect_data(reader)
    ranges = calibrater.compute_data()
    for name in ("input", "output"):
        minimum, maximum = ranges[name].range_value
        np.testing.assert_array_equal(minimum, [2.0])
        np.testing.assert_array_equal(maximum, [3.0])
