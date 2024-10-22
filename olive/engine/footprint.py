# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, DefaultDict, Dict, List, NamedTuple, Optional

from olive.common.config_utils import ConfigBase, config_json_dumps, config_json_loads
from olive.evaluator.metric_result import MetricResult

if TYPE_CHECKING:
    from olive.hardware import AcceleratorSpec


logger = logging.getLogger(__name__)


class RunHistory(NamedTuple):
    """Run history of a model."""

    model_id: str
    parent_model_id: str
    from_pass: str
    duration_sec: float
    metrics: str


class FootprintNodeMetric(ConfigBase):
    """Footprint Node metrics structure.

    value: {"metric_name": metrics_value, ...}
    cmp_direction: will be auto suggested. The format will be like: {"metric_name": 1, ...},
        1: higher is better, -1: lower is better
    if_goals_met: if the goals set by users are met
    """

    value: MetricResult = None
    cmp_direction: DefaultDict[str, int] = None
    if_goals_met: bool = False


class FootprintNode(ConfigBase):
    # None for no parent which means current model is the input model
    parent_model_id: str = None
    model_id: str
    model_config: Dict = None
    from_pass: str = None
    pass_run_config: Dict = None
    is_pareto_frontier: bool = False
    # TODO(trajep): add EP/accelerators for same_model_id metrics
    metrics: FootprintNodeMetric = None

    start_time: float = 0
    end_time: float = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Footprint:
    """The engine footprint is a class that contains the footprint of the engine runtime.

    It is used to collect the runtime state of the Olive engine and to organize the state
    in a way that is easy to visualize and understand for customers.
    """

    def __init__(
        self,
        nodes=None,
        objective_dict: Dict = None,
        is_marked_pareto_frontier: bool = False,
    ):
        self.nodes: Dict[str, FootprintNode] = nodes or OrderedDict()
        self.objective_dict = objective_dict or {}
        self.is_marked_pareto_frontier = is_marked_pareto_frontier

    def record_objective_dict(self, objective_dict):
        self.objective_dict = objective_dict

    def record(self, foot_print_node: FootprintNode = None, **kwargs):
        _model_id = kwargs.get("model_id")
        if foot_print_node is not None:
            _model_id = foot_print_node.model_id
            self.nodes[_model_id] = foot_print_node
        elif _model_id in self.nodes:
            self.nodes[_model_id].update(**kwargs)
        else:
            self.nodes[_model_id] = FootprintNode(**kwargs)
        self._resolve_metrics()

    def create_footprints_by_model_ids(self, model_ids) -> "Footprint":
        nodes = OrderedDict()
        for model_id in model_ids:
            # if model_id is not in self.nodes, KeyError will be raised
            nodes[model_id] = deepcopy(self.nodes[model_id])
        return Footprint(nodes=nodes, objective_dict=deepcopy(self.objective_dict))

    def create_pareto_frontier(self, output_model_num: int = None) -> Optional["Footprint"]:
        self._mark_pareto_frontier()
        if output_model_num is None or len(self.nodes) <= output_model_num:
            logger.info("Output all %d models", len(self.nodes))
            return self._create_pareto_frontier_from_nodes(self.nodes)
        else:
            topk_nodes = self.get_top_ranked_nodes(output_model_num)
            logger.info("Output top ranked %d models based on metric priorities", len(topk_nodes))
            return self._create_pareto_frontier_from_nodes(topk_nodes)

    def plot_pareto_frontier_to_html(self, index=None, save_path=None, is_show=False):
        self._plot_pareto_frontier(index, save_path, is_show, "html")

    def plot_pareto_frontier_to_image(self, index=None, save_path=None, is_show=False):
        self._plot_pareto_frontier(index, save_path, is_show, "image")

    def _create_pareto_frontier_from_nodes(self, nodes: Dict) -> Optional["Footprint"]:
        rls = {k: v for k, v in nodes.items() if v.is_pareto_frontier}
        if not rls:
            logger.warning("There is no pareto frontier points.")
            return None
        for v in rls.values():
            logger.info("pareto frontier points: %s \n%s", v.model_id, v.metrics.value)

        # restructure the pareto frontier points to instance of Footprints node for further analysis
        return Footprint(
            nodes=deepcopy(rls), objective_dict=deepcopy(self.objective_dict), is_marked_pareto_frontier=True
        )

    def summarize_run_history(self) -> List[RunHistory]:
        """Summarize the run history of a model.

        The summarization includes the columns of model_id, parent_model_id, from_pass, duration, metrics
        """
        rls = []
        for model_id, node in self.nodes.items():
            # get the run duration between current model and its parent model
            if node.parent_model_id is not None:
                duration = max(node.end_time - node.start_time, 0)
            else:
                duration = None
            run_history = RunHistory(
                model_id=model_id,
                parent_model_id=node.parent_model_id,
                from_pass=node.from_pass,
                duration_sec=duration,
                metrics=str(node.metrics.value) if node.metrics else None,
            )
            rls.append(run_history)
        return rls

    def trace_back_run_history(self, model_id) -> Dict[str, Dict]:
        """Trace back the run history of a model.

        The trace order: model_id -> parent_model_id1 -> parent_model_id2 -> ...
        """
        rls = OrderedDict()
        while model_id is not None:
            if model_id in rls:
                raise ValueError(f"Loop detected in the run history of model {model_id}")
            rls[model_id] = self.nodes[model_id].pass_run_config
            model_id = self.nodes[model_id].parent_model_id
        return rls

    def check_empty_nodes(self):
        return self.nodes is None or len(self.nodes) == 0

    def to_df(self):
        # to pandas.DataFrame
        raise NotImplementedError

    def to_json(self):
        return config_json_dumps(self.nodes)

    @classmethod
    def from_json(cls, json_str):
        nodes = OrderedDict()
        for k, v in config_json_loads(json_str, object_pairs_hook=OrderedDict).items():
            nodes[k] = FootprintNode(**v)
        return cls(nodes=nodes)

    def to_file(self, file_path):
        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, file_path):
        with open(file_path) as f:
            return cls.from_json(f.read())

    def get_output_model_id(self):
        # TODO(anyone): Make this more robust by ensuring there is only one pass flow and one output model
        return next(reversed(self.nodes.keys()))

    def get_output_model_path(self):
        return self.get_model_path(self.get_output_model_id())

    def get_model_config(self, model_id):
        model_config = self.nodes[model_id].model_config
        if model_config is None:
            return {}

        return model_config.get("config", {})

    def get_model_inference_config(self, model_id):
        return self.get_model_config(model_id).get("inference_settings", None)

    def get_model_path(self, model_id):
        return self.get_model_config(model_id).get("model_path", None)

    def get_model_type(self, model_id):
        model_config = self.nodes[model_id].model_config
        if model_config is None:
            return None

        return model_config.get("type", None)

    def get_use_ort_extensions(self, model_id):
        return self.get_model_config(model_id).get("use_ort_extensions", False)

    def get_input_node(self):
        return next(v for _, v in self.nodes.items() if v.parent_model_id is None)

    def _len_first_metric(self):
        if not self.nodes:
            return 0
        for metric in self.nodes.values():
            if not metric.metrics:
                continue
            return len(metric.metrics.value)
        return 0

    def _resolve_metrics(self):
        for k, v in self.nodes.items():
            if not v.metrics:
                continue
            if self.nodes[k].metrics.cmp_direction is None:
                self.nodes[k].metrics.cmp_direction = {}

            if_goals_met = []
            for metric_name in v.metrics.value:
                if metric_name not in self.objective_dict:
                    logger.debug("There is no goal set for metric: %s.", metric_name)
                    continue
                higher_is_better = self.objective_dict[metric_name]["higher_is_better"]
                cmp_direction = 1 if higher_is_better else -1
                self.nodes[k].metrics.cmp_direction[metric_name] = cmp_direction

                _goal = self.objective_dict[metric_name]["goal"]
                if _goal is None:
                    if_goals_met.append(True)
                else:
                    if_goals_met.append(
                        v.metrics.value[metric_name].value >= _goal
                        if cmp_direction == 1
                        else v.metrics.value[metric_name].value <= _goal
                    )
            self.nodes[k].metrics.if_goals_met = all(if_goals_met)

    def _get_candidates(self) -> Dict[str, FootprintNode]:
        candidates = {
            k: v
            for k, v in self.nodes.items()
            if v.metrics and v.parent_model_id is not None and v.metrics.if_goals_met
        }
        if not candidates:
            logger.warning(
                "There is no expected candidates. Please check: "
                "1. if the metric goal is too strict; "
                "2. if pass config is set correctly."
            )
        return candidates

    def _mark_pareto_frontier(self):
        if self.is_marked_pareto_frontier:
            return
        candidates = self._get_candidates()
        for k, v in candidates.items():
            # if current point's metrics is less than any other point's metrics, it is not pareto frontier
            cmp_flag = True
            for _k, _v in candidates.items():
                if k == _k:
                    # don't compare the point with itself
                    continue
                if not cmp_flag:
                    break
                # if all the metrics of current point is less than or equal to any other point's metrics
                # (i.e., it is dominated by the other point), it is not pareto frontier
                # e.g. current point's metrics is [1, 2, 3],
                # other point's metrics is [2, 3, 4], then current point is not pareto frontier
                # but if current point's metrics is [3, 2, 3], other point's metrics is [2, 3, 4],
                # then current point is pareto frontier
                # Note: equal points don't dominate one another
                equal = True  # two points are equal
                dominated = True  # current point is dominated by other point
                for metric_name in v.metrics.value:
                    if metric_name not in _v.metrics.cmp_direction:
                        logger.debug("Metric %s is not in cmp_direction, will not be compared.", metric_name)
                        continue
                    other_point_metrics = _v.metrics.value[metric_name].value * _v.metrics.cmp_direction[metric_name]
                    current_point_metrics = v.metrics.value[metric_name].value * v.metrics.cmp_direction[metric_name]
                    dominated &= current_point_metrics <= other_point_metrics
                    equal &= current_point_metrics == other_point_metrics
                # point is not on pareto frontier if dominated and not equal
                _against_pareto_frontier_check = dominated and not equal
                cmp_flag &= not _against_pareto_frontier_check
            self.nodes[k].is_pareto_frontier = cmp_flag
        self.is_marked_pareto_frontier = True

    def get_top_ranked_nodes(self, k: int) -> List[FootprintNode]:
        footprint_node_list = self.nodes.values()
        sorted_footprint_node_list = sorted(
            footprint_node_list,
            key=lambda x: tuple(
                (
                    x.metrics.value[metric].value
                    if x.metrics.cmp_direction[metric] == 1
                    else -x.metrics.value[metric].value
                )
                for metric in self.objective_dict
            ),
            reverse=True,
        )
        return sorted_footprint_node_list[:k]

    def _get_metrics_name_by_indices(self, indices) -> List[str]:
        """Get the first available metrics names by index."""
        for v in self.nodes.values():
            if v.metrics:
                rls = []
                for index in indices:
                    if isinstance(index, str):
                        if index in self.objective_dict:
                            rls.append(index)
                        else:
                            logger.error("the metric %s is not in the metrics", index)
                    elif isinstance(index, int):
                        if index < len(self.objective_dict):
                            rls.append(list(self.objective_dict.keys())[index])
                        else:
                            logger.error("the index %s is out of range", index)
                if rls:
                    return rls
        return []

    def _plot_pareto_frontier(self, ranks=None, save_path=None, is_show=True, save_format="html"):
        """Plot pareto frontier with plotly.

        :param ranks: the rank list of the metrics to be shown in the pareto frontier chart
        :param save_path: the path to save the pareto frontier chart
        :param is_show: whether to show the pareto frontier chart
        :param save_format: the format of the pareto frontier chart, can be "html" or "image"
        """
        assert save_path is not None or is_show, "you must specify the save path or set is_show to True"
        if self._len_first_metric() <= 1:
            logger.warning("There is no need to plot pareto frontier with only one metric")
            return

        ranks = ranks or [1, 2]
        index = [i - 1 for i in ranks]
        self._mark_pareto_frontier()
        nodes_to_be_plotted = self._get_candidates()

        if not nodes_to_be_plotted:
            logger.warning("there is no candidate to be plotted.")
            return
        # plot pareto frontier
        try:
            import pandas as pd
            import plotly.graph_objects as go

            pd.options.mode.chained_assignment = None
        except ImportError:
            logger.warning("Please make sure you installed pandas and plotly successfully.")
            return

        # select column to shown in pareto frontier chat
        metric_column = self._get_metrics_name_by_indices(index)
        # to support 3d pareto_frontier
        if len(metric_column) < 2:
            logger.error("you can not plot pareto frontier with less than 2 metrics")
            return
        dict_data = defaultdict(list)
        for k, v in nodes_to_be_plotted.items():
            dict_data["model_id"].append(k)
            dict_data["is_pareto_frontier"].append(v.is_pareto_frontier)
            dict_data["marker_color"].append("red" if v.is_pareto_frontier else "blue")
            dict_data["marker_size"].append(12 if v.is_pareto_frontier else 8)
            show_list = [k]
            for metric_name in metric_column:
                dict_data[metric_name].append(v.metrics.value[metric_name].value)
                show_list.append(f"{metric_name}: {v.metrics.value}")
            dict_data["show_text"].append("<br>".join(show_list))
        data = pd.DataFrame(dict_data)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dict_data[metric_column[index[0]]],
                y=dict_data[metric_column[index[1]]],
                mode="markers",
                name="all footprints",
                marker={"color": dict_data["marker_color"], "size": dict_data["marker_size"]},
                customdata=dict_data["show_text"],
                hovertemplate="%{customdata}",
            )
        )
        pareto_frontiers_data = data.loc[data["is_pareto_frontier"]]
        pareto_frontiers_data = pareto_frontiers_data.sort_values(metric_column, ascending=True)
        fig.add_trace(
            go.Scatter(
                x=pareto_frontiers_data[metric_column[index[0]]],
                y=pareto_frontiers_data[metric_column[index[1]]],
                mode="lines",
                name="pareto frontiers",
            )
        )
        size_params = {"width": 1000, "height": 600}
        fig.update_layout(xaxis_title=metric_column[index[0]], yaxis_title=metric_column[index[1]], **size_params)

        if save_path:
            if save_format == "html":
                save_path = f"{save_path}.html" if not str(save_path).endswith(".html") else save_path
                fig.write_html(save_path)
            elif save_format == "image":
                save_path = f"{save_path}.png" if not str(save_path).endswith(".png") else save_path
                fig.write_image(save_path)

        if is_show:
            fig.show()


def get_best_candidate_node(
    pf_footprints: Dict["AcceleratorSpec", Footprint], footprints: Dict["AcceleratorSpec", Footprint]
):
    """Select the best candidate node from the pareto frontier footprints.

    This function evaluates nodes from the given pareto frontier footprints and selects the top-ranked node
    based on specified objective metrics. It compares nodes from two dictionaries of footprints and
    ranks them according to their metrics.

    Args:
        pf_footprints (Dict["AcceleratorSpec", Footprint]): A dictionary mapping accelerator specifications
            to their corresponding pareto frontier footprints, which contain nodes and their metrics.
        footprints (Dict["AcceleratorSpec", Footprint"]): A dictionary mapping accelerator specifications
            to their corresponding footprints, which contain nodes and their metrics.

    Returns:
        Node: The top-ranked node based on the specified objective metrics.

    """
    objective_dict = next(iter(pf_footprints.values())).objective_dict
    top_nodes = []
    for accelerator_spec, pf_footprint in pf_footprints.items():
        footprint = footprints[accelerator_spec]
        if pf_footprint.nodes and footprint.nodes:
            top_nodes.append(next(iter(pf_footprint.get_top_ranked_nodes(1))))
    return next(
        iter(
            sorted(
                top_nodes,
                key=lambda x: tuple(
                    (
                        x.metrics.value[metric].value
                        if x.metrics.cmp_direction[metric] == 1
                        else -x.metrics.value[metric].value
                    )
                    for metric in objective_dict
                ),
                reverse=True,
            )
        )
    )
