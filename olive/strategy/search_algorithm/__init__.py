# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.strategy.search_algorithm.exhaustive import ExhaustiveSearchAlgorithm
from olive.strategy.search_algorithm.random_sampler import RandomSearchAlgorithm
from olive.strategy.search_algorithm.search_algorithm import SearchAlgorithm
from olive.strategy.search_algorithm.tpe_sampler import TPESearchAlgorithm

REGISTRY = SearchAlgorithm.registry

__all__ = ["SearchAlgorithm", "ExhaustiveSearchAlgorithm", "RandomSearchAlgorithm", "TPESearchAlgorithm"]
