# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from datetime import datetime
from collections import OrderedDict
import logging
from olive.common.config_utils import ConfigBase, config_json_dumps, config_json_loads

logger = logging.getLogger(__name__)


class FootprintNodeMetric(ConfigBase):
    metrics: dict
    cmp_direction: dict = {}
    is_goals_met: bool = False


class FootprintNode(ConfigBase):
    # None for no parent which means current model is the input model
    parent_model_id: str = None

    model_id: str
    from_pass: str
    config: dict = None
    is_pareto_frontier: bool = False
    metrics: FootprintNodeMetric = FootprintNodeMetric(
        metrics={}, is_goals_met=False
    )

    date_time: float = datetime.now().timestamp()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Footprint():
    """
    The engine footprint is a class that contains the footprint of the engine runtime.
    It is used to collect the runtime state of the Olive engine and to organize the state
    in a way that is easy to visualize and understand for customers.
    """

    def __init__(self,):
        self.footprints = OrderedDict()
        self.objective_dict = None

    def record_objective_dict(self, objective_dict):
        self.objective_dict = objective_dict

    def resolve_metrics(self):
        for k, v in self.footprints.items():
            if v.metrics is None:
                continue
            is_goals_met = []
            for metric_name in v.metrics.metrics:
                if metric_name in self.objective_dict:
                    cmp_direction = 1 if self.objective_dict[metric_name]["higher_is_better"] else -1
                    self.footprints[k].metrics.cmp_direction[metric_name] = cmp_direction
                    _goal = self.objective_dict[metric_name]["goal"]
                    if _goal is None:
                        is_goals_met.append(True)
                    else:
                        is_goals_met.append(
                            v.metrics.metrics[metric_name] * cmp_direction >= _goal
                        )
                else:
                    logger.warning(f"Metric {metric_name} is not in the objective dict")
            self.footprints[k].metrics.is_goals_met = all(is_goals_met)

    def record(self, foot_print_node: FootprintNode = None, **kwargs):
        _model_id = kwargs.get("model_id", None)
        if foot_print_node is not None:
            _model_id = foot_print_node.model_id
            self.footprints[_model_id] = foot_print_node
        elif _model_id in self.footprints:
            self.footprints[_model_id].update(**kwargs)
        else:
            self.footprints[_model_id] = FootprintNode(**kwargs)
        self.resolve_metrics()

    def get_candidates(self):
        return {k: v for k, v in self.footprints.items() if v.metrics is not None}

    def mark_pareto_frontier(self):
        for k, v in self.footprints.items():
            # if current point's metrics is less than any other point's metrics, it is not pareto frontier
            cmp_flag = True and v.metrics is not None and len(v.metrics.metrics) > 0
            for _, _v in self.footprints.items():
                if not cmp_flag:
                    break
                if _v.metrics is not None and len(_v.metrics.metrics) > 0:
                    _against_pareto_frontier_check = True
                    # if all the metrics of current point is less than any other point's metrics, it is not pareto frontier
                    # e.g. current point's metrics is [1, 2, 3], other point's metrics is [2, 3, 4], then current point is not pareto frontier
                    # but if current point's metrics is [3, 2, 3], other point's metrics is [2, 3, 4], then current point is pareto frontier
                    for metric_name in v.metrics.metrics:
                        other_point_metrics = _v.metrics.metrics[metric_name] * _v.metrics.cmp_direction[metric_name]
                        current_point_metrics = v.metrics.metrics[metric_name] * v.metrics.cmp_direction[metric_name]
                        _against_pareto_frontier_check &= current_point_metrics < other_point_metrics
                    cmp_flag &= not _against_pareto_frontier_check
            self.footprints[k].is_pareto_frontier = cmp_flag

    def get_pareto_frontier(self):
        self.mark_pareto_frontier()
        rls = {k: v for k, v in self.footprints.items() if v.is_pareto_frontier}
        for _, v in rls.items():
            logger.info(f"pareto frontier points: {v.model_id} {v.metrics.metrics}")
        return rls

    def plot_pareto_frontier(self):
        self.mark_pareto_frontier()
        # plot pareto frontier
        pass

    def to_df(self):
        # to pandas.DataFrame
        pass

    def to_json(self):
        return config_json_dumps(self.footprints)

    @classmethod
    def from_json(cls, json_str):
        footprint_obj = cls()
        for k, v in config_json_loads(json_str, object_pairs_hook=OrderedDict).items():
            footprint_obj.footprints[k] = FootprintNode(**v)
        return footprint_obj

    def to_file(self, file_path):
        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as f:
            return cls.from_json(f.read())
