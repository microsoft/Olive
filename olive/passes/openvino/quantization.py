# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from olive.common.user_module_loader import UserModuleLoader
from olive.model import OpenVINOModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class OpenVINOQuantization(Pass):
    """
    Post-training quantization for OpenVINO model.
    Please refer to https://docs.openvino.ai/latest/pot_introduction.html for more details.
    """

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "engine_config": PassConfigParam(
                type_=Dict,
                required=True,
                description=(
                    "Specific config for openvino.tools.pot.IEEngine. 'engine_config' can be set"
                    " by passing a dictonary, for example engine_config = {'device': 'CPU'}"
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
                type_=Union[Path, str],
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
                    " in the optimization pipeline is determined by the order in the list. example: algorithms = "
                    " [{'name': 'DefaultQuantization', 'params': {'preset': 'performance', 'stat_subset_size': 500},}]"
                ),
            ),
        }

    def _run_for_config(self, model: OpenVINOModel, config: Dict[str, Any], output_model_path: str) -> OpenVINOModel:

        try:
            from openvino.tools.pot import IEEngine, compress_model_weights, create_pipeline, save_model
        except ImportError:
            raise ImportError("Please install olive[openvino] to use OpenVINO model")

        model_name = model.name if model.name else "ov_model"

        loader = UserModuleLoader(user_script=config["user_script"], script_dir=config["script_dir"])
        data_loader = loader.call_object(config["dataloader_func"], config["data_dir"], config["batch_size"])
        metric = loader.load_object(config["metric_func"])
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
        openvino_model = OpenVINOModel(model_path, model_name)

        return openvino_model
