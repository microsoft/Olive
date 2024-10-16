# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import pytest

from olive.strategy.search_parameter import Categorical, CategoricalValue, Conditional, ConditionalDefault


def test_categorical_value():
    proxy = CategoricalValue()
    with pytest.raises(RuntimeError):
        proxy.get_support({})


def test_conditional_one_parent():
    conditional = Conditional(
        parents=("parent1",),
        support={("value1",): Categorical([1, 2, 3]), ("value2",): Categorical([4, 5, 6])},
        default=Categorical([7, 8, 9]),
    )
    assert conditional.get_support({"parent1": "value1"}) == [1, 2, 3]
    assert conditional.get_support({"parent1": "value2"}) == [4, 5, 6]
    assert conditional.get_support({"parent1": "value3"}) == [7, 8, 9]


def test_conditional_multiple_parents():
    conditional = Conditional(
        parents=("parent1", "parent2", "parent3"),
        support={
            ("value1", "value2", "value3"): Categorical([1, 2, 3]),
            ("value1", "value2", "value4"): Categorical([4, 5, 6]),
        },
        default=Categorical([7, 8, 9]),
    )
    assert conditional.get_support({"parent1": "value1", "parent2": "value2", "parent3": "value3"}) == [1, 2, 3]
    assert conditional.get_support({"parent1": "value1", "parent2": "value2", "parent3": "value4"}) == [4, 5, 6]
    assert conditional.get_support({"parent1": "value5", "parent2": "value6", "parent3": "value7"}) == [7, 8, 9]


def test_conditional_default():
    conditional = ConditionalDefault(
        parents=("parent1",), support={("value1",): [1, 2, 3], ("value2",): [4, 5, 6]}, default=[7, 8, 9]
    )
    assert conditional.get_support({"parent1": "value1"}) == [1, 2, 3]
    assert conditional.get_support({"parent1": "value2"}) == [4, 5, 6]
    assert conditional.get_support({"parent1": None}) == [7, 8, 9]
