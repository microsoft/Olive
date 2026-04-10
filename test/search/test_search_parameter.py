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
        # execute
        result = SpecialParamValue.IGNORED

        # assert
        assert result == "OLIVE_IGNORED_PARAM_VALUE"

    def test_invalid_value(self):
        # execute
        result = SpecialParamValue.INVALID

        # assert
        assert result == "OLIVE_INVALID_PARAM_VALUE"


class TestCategorical:
    def test_int_support(self):
        # setup
        cat = Categorical([1, 2, 3])

        # execute
        result = cat.get_support()

        # assert
        assert result == [1, 2, 3]

    def test_string_support(self):
        # setup
        cat = Categorical(["a", "b", "c"])

        # execute
        result = cat.get_support()

        # assert
        assert result == ["a", "b", "c"]

    def test_float_support(self):
        # setup
        cat = Categorical([0.1, 0.5, 1.0])

        # execute
        result = cat.get_support()

        # assert
        assert result == [0.1, 0.5, 1.0]

    def test_bool_support(self):
        # setup
        cat = Categorical([True, False])

        # execute
        result = cat.get_support()

        # assert
        assert result == [True, False]

    def test_repr(self):
        # setup
        cat = Categorical([1, 2, 3])

        # execute
        result = repr(cat)

        # assert
        assert result == "Categorical([1, 2, 3])"

    def test_to_json(self):
        # setup
        cat = Categorical([1, 2, 3])

        # execute
        result = cat.to_json()

        # assert
        assert result["olive_parameter_type"] == "SearchParameter"
        assert result["type"] == "Categorical"
        assert result["support"] == [1, 2, 3]


class TestBoolean:
    def test_support(self):
        # setup
        b = Boolean()

        # execute
        result = b.get_support()

        # assert
        assert result == [True, False]

    def test_is_categorical(self):
        # execute
        result = issubclass(Boolean, Categorical)

        # assert
        assert result


class TestConditional:
    def test_single_parent(self):
        # setup
        cond = Conditional(
            parents=("parent1",),
            support={
                ("value1",): Categorical([1, 2, 3]),
                ("value2",): Categorical([4, 5, 6]),
            },
            default=Categorical([7, 8, 9]),
        )

        # execute
        result_v1 = cond.get_support_with_args({"parent1": "value1"})
        result_v2 = cond.get_support_with_args({"parent1": "value2"})
        result_unknown = cond.get_support_with_args({"parent1": "unknown"})

        # assert
        assert result_v1 == [1, 2, 3]
        assert result_v2 == [4, 5, 6]
        assert result_unknown == [7, 8, 9]

    def test_multi_parent(self):
        # setup
        cond = Conditional(
            parents=("parent1", "parent2"),
            support={
                ("v1", "v2"): Categorical([10, 20]),
            },
        )

        # execute
        result = cond.get_support_with_args({"parent1": "v1", "parent2": "v2"})

        # assert
        assert result == [10, 20]

    def test_default_is_invalid(self):
        # setup
        cond = Conditional(
            parents=("p",),
            support={("a",): Categorical([1])},
        )

        # execute
        result = cond.get_support_with_args({"p": "missing"})

        # assert
        assert result == [SpecialParamValue.INVALID]

    def test_get_invalid_choice(self):
        # execute
        result = Conditional.get_invalid_choice()

        # assert
        assert isinstance(result, Categorical)
        assert result.get_support() == [SpecialParamValue.INVALID]

    def test_get_ignored_choice(self):
        # execute
        result = Conditional.get_ignored_choice()

        # assert
        assert isinstance(result, Categorical)
        assert result.get_support() == [SpecialParamValue.IGNORED]

    def test_repr(self):
        # setup
        cond = Conditional(
            parents=("p",),
            support={("a",): Categorical([1])},
        )

        # execute
        result = repr(cond)

        # assert
        assert "Conditional" in result
        assert "parents" in result

    def test_to_json(self):
        # setup
        cond = Conditional(
            parents=("p",),
            support={("a",): Categorical([1, 2])},
            default=Categorical([3]),
        )

        # execute
        result = cond.to_json()

        # assert
        assert result["olive_parameter_type"] == "SearchParameter"
        assert result["type"] == "Conditional"
        assert result["parents"] == ("p",)

    def test_condition_single_parent(self):
        # setup
        cond = Conditional(
            parents=("p",),
            support={
                ("a",): Categorical([1, 2]),
                ("b",): Categorical([3, 4]),
            },
        )

        # execute
        result = cond.condition({"p": "a"})

        # assert
        assert isinstance(result, Categorical)
        assert result.get_support() == [1, 2]

    def test_condition_returns_default_when_no_match(self):
        # setup
        cond = Conditional(
            parents=("p1", "p2"),
            support={("a", "b"): Categorical([1])},
            default=Categorical([99]),
        )

        # execute
        result = cond.condition({"p1": "missing"})

        # assert
        assert isinstance(result, Categorical)
        assert result.get_support() == [99]


class TestConditionalDefault:
    def test_basic(self):
        # setup
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1, ("b",): 2},
            default=3,
        )

        # execute
        result_a = cd.get_support_with_args({"p": "a"})
        result_b = cd.get_support_with_args({"p": "b"})
        result_c = cd.get_support_with_args({"p": "c"})

        # assert
        assert result_a == 1
        assert result_b == 2
        assert result_c == 3

    def test_default_invalid(self):
        # setup
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1},
        )

        # execute
        result = cd.get_support_with_args({"p": "missing"})

        # assert
        assert result == SpecialParamValue.INVALID

    def test_condition(self):
        # setup
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 10, ("b",): 20},
            default=30,
        )

        # execute
        result_a = cd.condition({"p": "a"})
        result_b = cd.condition({"p": "b"})
        result_c = cd.condition({"p": "c"})

        # assert
        assert result_a == 10
        assert result_b == 20
        assert result_c == 30

    def test_get_invalid_choice(self):
        # execute
        result = ConditionalDefault.get_invalid_choice()

        # assert
        assert result == SpecialParamValue.INVALID

    def test_get_ignored_choice(self):
        # execute
        result = ConditionalDefault.get_ignored_choice()

        # assert
        assert result == SpecialParamValue.IGNORED

    def test_to_json(self):
        # setup
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1},
            default=2,
        )

        # execute
        result = cd.to_json()

        # assert
        assert result["type"] == "ConditionalDefault"

    def test_repr(self):
        # setup
        cd = ConditionalDefault(
            parents=("p",),
            support={("a",): 1},
            default=2,
        )

        # execute
        result = repr(cd)

        # assert
        assert "ConditionalDefault" in result


class TestJsonToSearchParameter:
    def test_categorical(self):
        # setup
        json_data = {"olive_parameter_type": "SearchParameter", "type": "Categorical", "support": [1, 2, 3]}

        # execute
        result = json_to_search_parameter(json_data)

        # assert
        assert isinstance(result, Categorical)
        assert result.get_support() == [1, 2, 3]

    def test_unknown_type_raises(self):
        # setup
        json_data = {"olive_parameter_type": "SearchParameter", "type": "Unknown"}

        # execute & assert
        with pytest.raises(ValueError, match="Unknown search parameter type"):
            json_to_search_parameter(json_data)
