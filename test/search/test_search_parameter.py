# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from olive.search.search_parameter import (
    Boolean,
    Categorical,
    Conditional,
    ConditionalDefault,
    SpecialParamValue,
    json_to_search_parameter,
)


class TestSpecialParamValue:
    def test_ignored_value(self):
        assert SpecialParamValue.IGNORED == "OLIVE_IGNORED_PARAM_VALUE"

    def test_invalid_value(self):
        assert SpecialParamValue.INVALID == "OLIVE_INVALID_PARAM_VALUE"


class TestCategorical:
    def test_int_support(self):
        cat = Categorical([1, 2, 3])
        assert cat.get_support() == [1, 2, 3]

    def test_string_support(self):
        cat = Categorical(["a", "b", "c"])
        assert cat.get_support() == ["a", "b", "c"]

    def test_float_support(self):
        cat = Categorical([0.1, 0.5, 1.0])
        assert cat.get_support() == [0.1, 0.5, 1.0]

    def test_bool_support(self):
        cat = Categorical([True, False])
        assert cat.get_support() == [True, False]

    def test_repr(self):
        cat = Categorical([1, 2, 3])
        assert repr(cat) == "Categorical([1, 2, 3])"

    def test_to_json(self):
        cat = Categorical([1, 2, 3])
        result = cat.to_json()
        assert result["olive_parameter_type"] == "SearchParameter"
        assert result["type"] == "Categorical"
        assert result["support"] == [1, 2, 3]


class TestBoolean:
    def test_support(self):
        b = Boolean()
        assert b.get_support() == [True, False]

    def test_is_categorical(self):
        assert issubclass(Boolean, Categorical)


class TestConditional:
    def test_single_parent(self):
        cond = Conditional(
            parents=("parent1",),
            support={
                ("value1",): Categorical([1, 2, 3]),
                ("value2",): Categorical([4, 5, 6]),
            },
            default=Categorical([7, 8, 9]),
        )
        assert cond.get_support_with_args({"parent1": "value1"}) == [1, 2, 3]
        assert cond.get_support_with_args({"parent1": "value2"}) == [4, 5, 6]
        assert cond.get_support_with_args({"parent1": "unknown"}) == [7, 8, 9]

    def test_multi_parent(self):
        cond = Conditional(
            parents=("parent1", "parent2"),
            support={
                ("v1", "v2"): Categorical([10, 20]),
            },
        )
        assert cond.get_support_with_args({"parent1": "v1", "parent2": "v2"}) == [10, 20]

    def test_default_is_invalid(self):
        cond = Conditional(
            parents=("p",),
            support={("a",): Categorical([1])},
        )
        # Default is invalid choice
        assert cond.get_support_with_args({"p": "missing"}) == [SpecialParamValue.INVALID]

    def test_get_invalid_choice(self):
        result = Conditional.get_invalid_choice()
        assert isinstance(result, Categorical)
        assert result.get_support() == [SpecialParamValue.INVALID]

    def test_get_ignored_choice(self):
        result = Conditional.get_ignored_choice()
        assert isinstance(result, Categorical)
        assert result.get_support() == [SpecialParamValue.IGNORED]

    def test_repr(self):
        cond = Conditional(
            parents=("p",),
            support={("a",): Categorical([1])},
        )
        result = repr(cond)
        assert "Conditional" in result
        assert "parents" in result

    def test_to_json(self):
        cond = Conditional(
            parents=("p",),
            support={("a",): Categorical([1, 2])},
            default=Categorical([3]),
        )
        result = cond.to_json()
        assert result["olive_parameter_type"] == "SearchParameter"
        assert result["type"] == "Conditional"
        assert result["parents"] == ("p",)

    def test_condition_single_parent(self):
        cond = Conditional(
            parents=("p",),
            support={
                ("a",): Categorical([1, 2]),
                ("b",): Categorical([3, 4]),
            },
        )
        result = cond.condition({"p": "a"})
        assert isinstance(result, Categorical)
        assert result.get_support() == [1, 2]

    def test_condition_returns_default_when_no_match(self):
        cond = Conditional(
            parents=("p1", "p2"),
            support={("a", "b"): Categorical([1])},
            default=Categorical([99]),
        )
        result = cond.condition({"p1": "missing"})
        assert isinstance(result, Categorical)
        assert result.get_support() == [99]


class TestConditionalDefault:
    def test_basic(self):
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1, ("b",): 2},
            default=3,
        )
        assert cd.get_support_with_args({"p": "a"}) == 1
        assert cd.get_support_with_args({"p": "b"}) == 2
        assert cd.get_support_with_args({"p": "c"}) == 3

    def test_default_invalid(self):
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1},
        )
        assert cd.get_support_with_args({"p": "missing"}) == SpecialParamValue.INVALID

    def test_condition(self):
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 10, ("b",): 20},
            default=30,
        )
        assert cd.condition({"p": "a"}) == 10
        assert cd.condition({"p": "b"}) == 20
        assert cd.condition({"p": "c"}) == 30

    def test_get_invalid_choice(self):
        assert ConditionalDefault.get_invalid_choice() == SpecialParamValue.INVALID

    def test_get_ignored_choice(self):
        assert ConditionalDefault.get_ignored_choice() == SpecialParamValue.IGNORED

    def test_to_json(self):
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1},
            default=2,
        )
        result = cd.to_json()
        assert result["type"] == "ConditionalDefault"

    def test_repr(self):
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1},
            default=2,
        )
        result = repr(cd)
        assert "ConditionalDefault" in result


class TestJsonToSearchParameter:
    def test_categorical(self):
        json_data = {"olive_parameter_type": "SearchParameter", "type": "Categorical", "support": [1, 2, 3]}
        result = json_to_search_parameter(json_data)
        assert isinstance(result, Categorical)
        assert result.get_support() == [1, 2, 3]

    def test_unknown_type_raises(self):
        json_data = {"olive_parameter_type": "SearchParameter", "type": "Unknown"}
        with pytest.raises(ValueError, match="Unknown search parameter type"):
            json_to_search_parameter(json_data)
