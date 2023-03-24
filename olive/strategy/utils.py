# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List, Set, Tuple

import numpy as np

from olive.strategy.search_parameter import Conditional, SearchParameter


class DirectedGraph:
    def __init__(self, vertices: List[str], edges: List[Tuple[str, str]] = None):
        self.vertices = vertices
        self.graph = {v: [] for v in vertices}
        edges = edges or []
        for v1, v2 in edges:
            self.add_edge(v1, v2)

    def add_edge(self, v1: str, v2: str):
        assert v1 in self.vertices and v2 in self.vertices
        self.graph[v1].append(v2)

    def _is_cyclic_util(self, v: str, visited: Set[str], rec_stack: Set[str]):
        visited.add(v)
        rec_stack.add(v)

        for neighbor in self.graph[v]:
            if neighbor not in visited:
                if self._is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(v)
        return False

    def is_cyclic(self):
        visited = set()
        rec_stack = set()

        for v in self.vertices:
            if v not in visited:
                if self._is_cyclic_util(v, visited, rec_stack):
                    return True

        return False

    def _topological_sort_util(self, v: str, visited: Set[str], order: List[str]):
        visited.add(v)

        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self._topological_sort_util(neighbor, visited, order)

        order.insert(0, v)

    def topological_sort(self):
        assert not self.is_cyclic(), "Graph is cyclic, cannot perform topological sort."
        visited = set()
        order = []

        for v in self.vertices:
            if v not in visited:
                self._topological_sort_util(v, visited, order)

        return order


def _search_space_graph(search_space: Dict[str, SearchParameter]) -> DirectedGraph:
    """
    Create a directed graph from the search space.
    """
    graph = DirectedGraph(list(search_space.keys()))
    for name, param in search_space.items():
        if isinstance(param, Conditional):
            for parent in param.parents:
                graph.add_edge(parent, name)
    return graph


def cyclic_search_space(search_space: Dict[str, SearchParameter]) -> bool:
    """
    Check if the search space is cyclic.
    """
    graph = _search_space_graph(search_space)
    return graph.is_cyclic()


def order_search_parameters(search_space: Dict[str, SearchParameter]) -> List[str]:
    """
    Order the search parameters in a topological order.
    """
    graph = _search_space_graph(search_space)
    return graph.topological_sort()


def find_pareto_frontier_points(points: np.ndarray) -> List[int]:
    """
    Find the pareto frontier points in a set of points.
    """
    assert points.ndim == 2, "Points must be a 2D array."
    # assert points.shape[1] > 1, "Points must have more than 1 dimension."

    # Find the pareto frontier points
    pareto_frontier_points = np.array([])
    for idx, point in enumerate(points):
        if len(pareto_frontier_points) == 0:
            pareto_frontier_points = np.append(pareto_frontier_points, idx)
            continue

        frontier = points[pareto_frontier_points.astype(int)]

        # check if point is dominated by another point on the pareto front
        is_dominated = np.all(point <= frontier, axis=1)
        if np.any(is_dominated):
            continue

        # remove points from the pareto front dominated by point
        dominates = np.all(point >= frontier, axis=1)
        pareto_frontier_points = pareto_frontier_points[~dominates]

        # add point to the pareto front
        pareto_frontier_points = np.append(pareto_frontier_points, idx)

    return pareto_frontier_points.astype(int).tolist()
