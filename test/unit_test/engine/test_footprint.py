# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import tempfile
from pathlib import Path

import pytest

from olive.engine.footprint import Footprint


class TestFootprint:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.footprint_file = Path(__file__).parent / "mock_data" / "footprints.json"
        self.fp = Footprint.from_file(self.footprint_file)
        self.input_node = {k: v for k, v in self.fp.nodes.items() if v.parent_model_id is None}

    def test_create_from_model_ids(self):
        new_fp = self.fp.create_footprints_by_model_ids(self.fp.nodes.keys())
        assert len(new_fp.nodes) == len(self.fp.nodes)
        assert new_fp.nodes == self.fp.nodes
        assert new_fp.nodes is not self.fp.nodes
        assert new_fp.objective_dict == self.fp.objective_dict
        assert new_fp.objective_dict is not self.fp.objective_dict

    def test_file_dump(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.fp.to_file(Path(tempdir) / "footprint.json")
            fp2 = Footprint.from_file(Path(tempdir) / "footprint.json")
            assert len(fp2.nodes) == 3

    def test_json_dump(self):
        json_fp = self.fp.to_json()
        fp2 = Footprint.from_json(json_fp)
        assert len(fp2.nodes) == 3

    def test_pareto_frontier(self):
        pareto_frontier_fp = self.fp.create_pareto_frontier()
        assert isinstance(pareto_frontier_fp, Footprint)
        assert len(pareto_frontier_fp.nodes) == 2
        assert all(v.is_pareto_frontier for v in pareto_frontier_fp.nodes.values())

    def test_trace_back_run_history(self):
        for model_id in self.fp.nodes:
            run_history = self.fp.trace_back_run_history(model_id)
            assert run_history is not None
            assert next(reversed(run_history)) in self.input_node

    def test_get_model_info(self):
        for model_id in self.fp.nodes:
            inference_config = self.fp.get_model_inference_config(model_id)
            model_path = self.fp.get_model_path(model_id)
            if model_id in self.input_node:
                assert inference_config is None
                assert str(model_path).endswith(".pt")
            else:
                assert inference_config is not None
                assert str(model_path).endswith(".onnx")

    def test_plot_pareto_frontier(self):
        with tempfile.TemporaryDirectory() as tempdir:
            self.fp.objective_dict = {
                "accuracy-accuracy_score": {"higher_is_better": True, "priority": 1},
                "latency-avg": {"higher_is_better": False, "priority": 2},
            }
            self.fp.plot_pareto_frontier(
                is_show=False,
                save_path=Path(tempdir) / "pareto_frontier.html",
            )
            assert (Path(tempdir) / "pareto_frontier.html").exists()
