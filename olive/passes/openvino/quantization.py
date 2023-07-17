# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy as np

from olive.cache import get_local_path
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OpenVINOModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS


class OpenVINOQuantization(Pass):
    """
    Post-training quantization for OpenVINO model.
    Please refer to https://docs.openvino.ai/latest/pot_introduction.html for more details.
    """

    _requires_user_script = True
    _requires_data_config = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "engine_config": PassConfigParam(
                type_=Dict,
                required=True,
                description=(
                    "Specific config for openvino.tools.pot.IEEngine. 'engine_config' can be set"
                    " by passing a dictionary, for example engine_config: {'device': 'CPU'}"
                ),
            ),
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                is_object=True,
                description=(
                    "A callable function or a str of the function name from 'user_script'"
                    " for the instance of the dataloader."
                ),
            ),
            "data_dir": PassConfigParam(
                type_=OLIVE_RESOURCE_ANNOTATIONS,
                is_path=True,
                description="Dataset path. 'data_dir' can be by a str or Pathlib.Path.",
            ),
            "batch_size": PassConfigParam(type_=int, default_value=1, description="Batch size for the dataloader."),
            "metric_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                is_object=True,
                description=(
                    "A callable function or a str of the function name from 'user_script'"
                    " for Metric instance to calculate the accuracy metric of the model."
                ),
            ),
            "algorithms": PassConfigParam(
                type_=List[Dict],
                required=True,
                description=(
                    "A list defining optimization algorithms and their parameters included"
                    " in the optimization pipeline. The order in which they are applied to the model"
                    " in the optimization pipeline is determined by the order in the list. example: algorithms: "
                    " [{'name': 'DefaultQuantization', 'params': {'preset': 'performance', 'stat_subset_size': 500},}]"
                ),
            ),
        }

    def _run_for_config(self, model: OpenVINOModel, config: Dict[str, Any], output_model_path: str) -> OpenVINOModel:
        try:
            from openvino.tools.pot import IEEngine, compress_model_weights, create_pipeline, save_model
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model")

        # output model always has ov_model name stem
        model_name = "ov_model"

        if config["dataloader_func"]:
            data_loader = self._user_module_loader.call_object(
                config["dataloader_func"], get_local_path(config["data_dir"]), config["batch_size"]
            )
        elif self._data_config:
            common_dataloader = self._data_config.to_data_container().create_dataloader()
            data_loader = self._create_dataloader(common_dataloader)

        metric = self._user_module_loader.load_object(config["metric_func"])
        engine = IEEngine(config=config["engine_config"], data_loader=data_loader, metric=metric)
        self.pipeline = create_pipeline(config["algorithms"], engine)

        compressed_model = self.pipeline.run(model=model.load_model())
        compress_model_weights(compressed_model)
        compressed_model_paths = save_model(
            model=compressed_model,
            save_path=output_model_path,
            model_name=model_name,
        )
        model_path = Path(compressed_model_paths[0]["model"]).parent
        openvino_model = OpenVINOModel(model_path)

        return openvino_model

    def _create_dataloader(self, common_dataloader):
        """
        Create an openvino.tools.pot.api.DataLoader instance from a common dataloader.
        """
        try:
            from openvino.tools.pot.api import DataLoader
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO pass")

        class _OVDataloader(DataLoader):
            def __init__(self, dataloader):
                self.data = []
                self.labels = []
                for data, label in dataloader:
                    if isinstance(data, dict):
                        data = {k: np.array(v) for k, v in data.items()}
                    elif isinstance(data, tuple):
                        data = tuple(np.array(v) for v in data)
                    else:
                        data = np.array(data)
                    self.data.append(data)
                    self.labels.append(label)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                if index >= len(self):
                    raise IndexError

                return self.data[index], self.labels[index]

        return _OVDataloader(common_dataloader)
