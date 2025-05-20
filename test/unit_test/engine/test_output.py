# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from olive.engine.footprint import Footprint, FootprintNode, FootprintNodeMetric
from olive.engine.output import DeviceOutput, ModelOutput, WorkflowOutput
from olive.evaluator.metric_result import MetricResult
from olive.hardware.accelerator import AcceleratorSpec, Device, ExecutionProvider

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


class TestDeviceOutput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # setup
        self.device = Device.CPU
        self.objective_dict = {"accuracy": {"higher_is_better": True, "goal": 0.88, "priority": 1}}
        self.model_id_1 = "model1"
        self.model_id_2 = "model2"
        self.model_id_3 = "model3"

        # Create sample nodes with different metrics
        self.node1 = create_node(self.model_id_1, {"accuracy": 95})
        self.node2 = create_node(self.model_id_2, {"accuracy": 90})
        self.node3 = create_node(self.model_id_3, {"accuracy": 98})

        # Create footprints
        self.footprint_cpu = Footprint(nodes={self.model_id_1: self.node1, self.model_id_2: self.node2})
        self.footprint_cuda = Footprint(nodes={self.model_id_3: self.node3})

        # Create ep_footprint_map
        self.ep_footprint_map = {
            "CPUExecutionProvider": self.footprint_cpu,
            "OpenVINOExecutionProvider": self.footprint_cuda,
        }

        self.acc_output = DeviceOutput(self.device, self.ep_footprint_map, self.objective_dict)

    def test_has_output_model(self, setup):
        # assert
        assert self.acc_output.has_output_model()

    def test_empty_footprint(self):
        # setup
        empty_footprint = Footprint(nodes={})

        # execute
        acc_output = DeviceOutput(self.device, {"CPUExecutionProvider": empty_footprint}, self.objective_dict)

        # assert
        assert not acc_output.has_output_model()

    def test_get_output_models(self):
        # execute
        models = self.acc_output.get_output_models()

        # assert
        assert len(models) == 3
        model_ids = [model.model_id for model in models]
        assert self.model_id_1 in model_ids
        assert self.model_id_2 in model_ids
        assert self.model_id_3 in model_ids

    def test_getitem(self):
        # execute
        cpu_models = self.acc_output["CPUExecutionProvider"]

        # assert
        assert len(cpu_models) == 2

        cuda_models = self.acc_output["OpenVINOExecutionProvider"]
        assert len(cuda_models) == 1

    def test_get_best_candidate(self):
        # execute
        best = self.acc_output.get_best_candidate()

        # assert
        assert best is not None
        assert best.model_id == self.model_id_3

    def test_get_best_candidate_by_execution_provider(self):
        # execute
        best = self.acc_output.get_best_candidate_by_execution_provider(ExecutionProvider.CPUExecutionProvider)

        # assert
        assert best is not None
        assert best.model_id == self.model_id_1


class TestWorkflowOutput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # setup
        self.input_model_id = "input_model"
        self.node_cpu_1_id = "model_cpu_1"
        self.node_cpu_2_id = "model_cpu_2"
        self.node_gpu_1_id = "model_gpu_1"
        self.node_gpu_2_id = "model_gpu_2"

        self.input_node = create_node(self.input_model_id, {"accuracy": 85})
        self.node_cpu_1 = create_node(self.node_cpu_1_id, {"accuracy": 90})
        self.node_cpu_2 = create_node(self.node_cpu_2_id, {"accuracy": 88})
        self.node_gpu_1 = create_node(self.node_gpu_1_id, {"accuracy": 87})
        self.node_gpu_2 = create_node(self.node_gpu_2_id, {"accuracy": 89})

        # Create footprints
        self.footprint_cpu = Footprint(
            nodes={self.node_cpu_1.model_id: self.node_cpu_1, self.node_cpu_2.model_id: self.node_cpu_2}
        )
        objective_dict = {"accuracy": {"higher_is_better": True, "goal": 0.88, "priority": 1}}
        self.footprint_cpu.input_model_id = self.input_model_id
        self.footprint_cpu.objective_dict = objective_dict

        self.footprint_gpu = Footprint(
            nodes={self.node_gpu_1.model_id: self.node_gpu_1, self.node_gpu_2.model_id: self.node_gpu_2}
        )
        self.footprint_gpu.input_model_id = self.input_model_id
        self.footprint_gpu.objective_dict = objective_dict

        # Create output_acc_footprint_map and all_footprints
        self.acc_spec_cpu = AcceleratorSpec(Device.CPU, "CPUExecutionProvider")
        self.acc_spec_gpu = AcceleratorSpec(Device.GPU, "CUDAExecutionProvider")

        self.output_acc_footprint_map = {self.acc_spec_cpu: self.footprint_cpu, self.acc_spec_gpu: self.footprint_gpu}
        all_footprints_cpu = deepcopy(self.footprint_cpu)
        all_footprints_cpu.nodes[self.input_model_id] = self.input_node
        all_footprints_gpu = deepcopy(self.footprint_gpu)
        all_footprints_gpu.nodes[self.input_model_id] = self.input_node
        self.all_footprints = {self.acc_spec_cpu: all_footprints_cpu, self.acc_spec_gpu: all_footprints_gpu}

        # execute
        self.workflow_output = WorkflowOutput(self.output_acc_footprint_map, self.all_footprints)

    def test_get_input_model_metrics(self):
        # execute
        metrics = self.workflow_output.get_input_model_metrics()

        # assert
        assert metrics == {"accuracy": {"higher_is_better": True, "priority": 1, "value": 85}}

    def test_getitem(self):
        # assert
        cpu_output = self.workflow_output[Device.CPU.value]
        assert cpu_output is not None

        gpu_output = self.workflow_output[Device.GPU.value]
        assert gpu_output is not None

        # Test with non-existent device
        assert self.workflow_output["TPU"] is None

    def test_available_devices(self):
        # execute
        devices = self.workflow_output.get_available_devices()

        # assert
        assert len(devices) == 2
        assert Device.CPU.value in devices
        assert Device.GPU.value in devices

    def test_has_output_model(self):
        # assert
        assert self.workflow_output.has_output_model()

    def test_get_output_models_by_device(self):
        # execute
        cpu_models = self.workflow_output.get_output_models_by_device(Device.CPU)
        gpu_models = self.workflow_output.get_output_models_by_device(Device.GPU)
        cpu_models_str = self.workflow_output.get_output_models_by_device("CPU")

        # assert
        assert len(cpu_models) == 2
        assert len(gpu_models) == 2
        assert len(cpu_models_str) == 2

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
        assert len(models) == 4

    def test_get_best_candidate_by_device(self):
        # execute
        best_cpu = self.workflow_output.get_best_candidate_by_device(Device.CPU)
        best_gpu = self.workflow_output.get_best_candidate_by_device(Device.GPU)

        # assert
        assert best_cpu.model_id == self.node_cpu_1_id
        assert best_gpu.model_id == self.node_gpu_2_id

    def test_get_best_candidate(self):
        # execute
        best = self.workflow_output.get_best_candidate()

        # assert
        assert best.model_id == self.node_cpu_1_id

    def test_empty_workflow_output(self):
        # setup
        empty_footprint = Footprint()

        # execute
        empty_workflow = WorkflowOutput({}, {self.acc_spec_cpu: empty_footprint})

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
