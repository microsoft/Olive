# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict

from olive.search.search_parameter import SpecialParamValue
from olive.search.search_point import SearchPoint
from olive.search.search_sample import SearchSample


class TestSearchSample:
    def _make_sample(self, invalid_param=False, ignored_param=False):
        param_value = "a"
        if invalid_param:
            param_value = SpecialParamValue.INVALID
        elif ignored_param:
            param_value = SpecialParamValue.IGNORED

        values = OrderedDict(
            {
                "pass1": (
                    0,
                    OrderedDict(
                        {
                            "param1": (0, param_value),
                            "param2": (1, 10),
                        }
                    ),
                )
            }
        )
        point = SearchPoint(index=0, values=values)
        return SearchSample(search_point=point, model_ids=["model_0"])

    def test_creation(self):
        sample = self._make_sample()
        assert sample.model_ids == ["model_0"]
        assert sample.search_point.index == 0

    def test_passes_configs_valid(self):
        sample = self._make_sample()
        configs = sample.passes_configs
        assert configs is not None
        assert "pass1" in configs
        assert configs["pass1"]["params"]["param1"] == "a"
        assert configs["pass1"]["params"]["param2"] == 10

    def test_passes_configs_with_invalid_returns_none(self):
        sample = self._make_sample(invalid_param=True)
        assert sample.passes_configs is None

    def test_passes_configs_with_ignored_excludes_param(self):
        sample = self._make_sample(ignored_param=True)
        configs = sample.passes_configs
        assert configs is not None
        assert "param1" not in configs["pass1"]["params"]
        assert configs["pass1"]["params"]["param2"] == 10

    def test_to_json(self):
        sample = self._make_sample()
        result = sample.to_json()
        assert "search_point" in result
        assert "model_ids" in result
        assert result["model_ids"] == ["model_0"]

    def test_from_json_roundtrip(self):
        sample = self._make_sample()
        json_data = sample.to_json()
        restored = SearchSample.from_json(json_data)
        assert restored.model_ids == sample.model_ids
        assert restored.search_point.index == sample.search_point.index

    def test_repr(self):
        sample = self._make_sample()
        result = repr(sample)
        assert "SearchSample" in result
