# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OliveModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.strategy.search_parameter import Categorical

logger = logging.getLogger(__name__)


# static quantization specific config
_dataloader_config = {
    "batch_size": PassConfigParam(
        type_=int,
        default_value=1,
        description="Batch size for calibration.",
    ),
    "calib_size": PassConfigParam(
        type_=int,
        default_value=64,
        description="Calibration data size.",
    ),
    "dataloader_func": PassConfigParam(
        type_=Union[Callable, str],
        category=ParamCategory.OBJECT,
        required=True,
        description="Function/function name to generate dataloader for calibration.",
    ),
    "dataloader_func_kwargs": PassConfigParam(
        type_=Dict[str, Any],
        description="Keyword arguments for dataloader_func.",
    ),
}


class NVModelOptQuantization(Pass):
    """Quantize ONNX model with Nvidia-ModelOpt."""

    # set this to True if the pass has parameters that are functions or objects and the user script is required
    # to import the module containing the function or object
    _requires_user_script: bool = True

    class Precision(str, Enum):
        FP8 = "fp8"
        INT8 = "int8"
        INT4 = "int4"

        def __str__(self) -> str:
            return self.value

    class Algorithm(str, Enum):
        RTN = "RTN"
        AWQ = "AWQ"

        def __str__(self) -> str:
            return str(self.value)

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "precision": PassConfigParam(
                type_=NVModelOptQuantization.Precision,
                default_value="int4",
                searchable_values=Categorical(["fp8", "int8", "int4"]),
                description="NVModelOpt Quantization mode.",
            ),
            "algorithm": PassConfigParam(
                type_=NVModelOptQuantization.Algorithm,
                default_value="RTN",
                searchable_values=Categorical(["RTN", "AWQ"]),
                description="Algorithm of weight only quantization. Support 'RTN' and 'AWQ'.",
            ),
        }

        config.update(deepcopy(_dataloader_config))
        return config

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})

        if search_point["precision"] != NVModelOptQuantization.Precision.INT4 or search_point["algorithm"] not in [
            NVModelOptQuantization.Algorithm.RTN,
            NVModelOptQuantization.Algorithm.AWQ,
        ]:
            logger.error("Only INT4 quantization with RTN and AWQ algorithm is supported.")
            return False

        return True

    def _run_for_config(
        self, model: OliveModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:
        try:
            from modelopt.onnx.quantization.int4 import quantize_int4  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Please install `olive-ai[nvmo]` or `nvidia-modelopt` to use INT4 AWQ quantization!"
            ) from exc

        calib_dataloader = self._user_module_loader.call_object(
            config["dataloader_func"],
            config["batch_size"],
            config["calib_size"],
            model_path=model.model_path,
            **(config["dataloader_func_kwargs"] or {}),
        )

        quantize_mode = (
            "int4_awq_clip" if config["algorithm"] == NVModelOptQuantization.Algorithm.AWQ else "int4_rtn_dq"
        )
        q_model = quantize_int4(quantize_mode, model.load_model(), calib_dataloader)

        # save the model to the output path and return the model
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        return model_proto_to_olive_model(q_model, output_model_path, config)
