# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest
from pathlib import Path
import tempfile

from olive.engine.footprint import Footprint


class TestFootprint:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.footprint_file = Path(__file__).parent / "mock_data" / "footprints.json"
        self.fp = Footprint.from_file(self.footprint_file)

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
        pareto_frontier_fp = self.fp.get_pareto_frontier()
        assert isinstance(pareto_frontier_fp, Footprint)
        assert len(pareto_frontier_fp.nodes) == 2
        assert all([v.is_pareto_frontier for v in pareto_frontier_fp.nodes.values()])
