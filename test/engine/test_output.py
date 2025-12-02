# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.engine.footprint import Footprint, FootprintNode, FootprintNodeMetric
from olive.engine.output import ModelOutput, WorkflowOutput
from olive.evaluator.metric_result import MetricResult
from olive.hardware.accelerator import AcceleratorSpec, Device

# pylint: disable=W0201, W0212


def test_model_output():
    # setup
    model_id = "test_model_id"
    parent_model_id = "parent_model_id"
    from_pass = "test_pass"
    model_path = "path/to/model.onnx"
    model_type = "onnxmodel"
    metrics_value = {"latency": 10}
    metrics_json = {"value": metrics_value, "cmp_direction": {"latency": 1}}
    metrics_value_json = {"latency": {"value": 10}}
    metrics_mock = MagicMock()
    metrics_mock.to_json.return_value = metrics_json

    metrics_value_mock = MagicMock()
    metrics_value_mock.to_json.return_value = metrics_value_json
    metrics_mock.value = metrics_value_mock
    inference_settings = {"inference": "settings"}

    node = FootprintNode(
        model_id=model_id,
        parent_model_id=parent_model_id,
        from_pass=from_pass,
        metrics=metrics_mock,
        model_config={
            "type": model_type,
            "config": {"model_path": model_path, "inference_settings": inference_settings, "use_ort_extension": True},
        },
    )
    node.metrics = metrics_mock
    device = Device.CPU
    ep = "CPUExecutionProvider"

    # execute
    model_output = ModelOutput(device, ep, node)

    # assert
    assert model_output.metrics == metrics_json
    assert model_output.metrics_value == metrics_value_json
    assert model_output.model_path == model_path
    assert model_output.model_id == model_id
    assert model_output.model_type == model_type
    assert model_output.from_device() == device
    assert model_output.from_execution_provider() == ep
    assert model_output.from_pass() == from_pass
    assert model_output.get_parent_model_id() == parent_model_id
    assert model_output.use_ort_extension()
    assert model_output.get_inference_config() == inference_settings


def test_empty_node_raises_error():
    with pytest.raises(ValueError, match="FootprintNode cannot be None."):
        ModelOutput(Device.CPU, "CPUExecutionProvider", None)


class TestWorkflowOutput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # setup
        self.input_model_id = "input_model"
        self.node_cpu_1_id = "model_cpu_1"
        self.node_cpu_2_id = "model_cpu_2"

        self.input_node = create_node(self.input_model_id, {"accuracy": 85})
        self.node_cpu_1 = create_node(self.node_cpu_1_id, {"accuracy": 90})
        self.node_cpu_2 = create_node(self.node_cpu_2_id, {"accuracy": 88})

        # Create footprints
        footprint = Footprint(
            nodes={
                self.input_model_id: self.input_node,
                self.node_cpu_1.model_id: self.node_cpu_1,
                self.node_cpu_2.model_id: self.node_cpu_2,
            }
        )
        footprint.output_model_ids = [self.node_cpu_1_id, self.node_cpu_2_id]
        objective_dict = {"accuracy": {"higher_is_better": True, "goal": 0.88, "priority": 1}}
        footprint.input_model_id = self.input_model_id
        footprint.objective_dict = objective_dict

        # Create output_acc_footprint_map and all_footprints
        self.acc_spec = AcceleratorSpec(Device.CPU, "CPUExecutionProvider")

        # execute
        self.workflow_output = WorkflowOutput(self.acc_spec, footprint)
        self.footprint = footprint

    def test_get_input_model_metrics(self):
        # execute
        metrics = self.workflow_output.get_input_model_metrics()

        # assert
        assert metrics == {"accuracy": {"higher_is_better": True, "priority": 1, "value": 85}}

    def test_has_output_model(self):
        # assert
        assert self.workflow_output.has_output_model()

    def test_get_output_model_by_id(self):
        # execute
        model = self.workflow_output.get_output_model_by_id(self.node_cpu_1_id)

        # assert
        assert model is not None
        assert model.model_id == self.node_cpu_1_id

        # Test with non-existent ID
        model = self.workflow_output.get_output_model_by_id("non_existent")
        assert model is None

    def test_get_output_models(self):
        # execute
        models = self.workflow_output.get_output_models()

        # assert
        assert len(models) == 2

    def test_get_best_candidate(self):
        # execute
        best = self.workflow_output.get_best_candidate()

        # assert
        assert best.model_id == self.node_cpu_1_id

    def test_empty_workflow_output(self):
        # setup
        empty_footprint = self.footprint
        empty_footprint.output_model_ids = []
        empty_footprint.objective_dict = {}

        # execute
        empty_workflow = WorkflowOutput(self.acc_spec, empty_footprint)

        # assert
        assert empty_workflow._objective_dict == {}
        assert empty_workflow.get_best_candidate() is None
        assert empty_workflow.get_output_models() == []


def create_node(model_id, metric_values):
    metrics = FootprintNodeMetric()
    metric_dict = {}
    for metric_name, value in metric_values.items():
        metric_dict[metric_name] = {"value": value, "priority": 1, "higher_is_better": True}

    metrics.value = MetricResult(__root__=metric_dict)
    metrics.cmp_direction = dict.fromkeys(metric_values, 1)

    return FootprintNode(
        model_id=model_id,
        model_config={"type": "onnxmodel", "config": {"model_path": f"path/to/{model_id}.onnx"}},
        metrics=metrics,
    )
