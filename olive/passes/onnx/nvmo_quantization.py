# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OliveModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Categorical

logger = logging.getLogger(__name__)


# static quantization specific config
_dataloader_config = {
    "data_config": PassConfigParam(
        type_=Union[DataConfig, Dict],
        required=True,
        description="Data config to load data for computing latency.",
    ),
}


class NVModelOptQuantization(Pass):
    """Quantize ONNX model with Nvidia-ModelOpt."""

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
        return {
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
            **deepcopy(_dataloader_config),
        }

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
        self, model: OliveModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:
        try:
            from modelopt.onnx.quantization.int4 import quantize_int4  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Please install `olive-ai[nvmo]` or `nvidia-modelopt[onnx]` to use INT4 AWQ quantization!"
            ) from exc

        data_config = validate_config(config["data_config"], DataConfig)
        calib_dataloader = data_config.to_data_container().create_dataloader()

        quantize_mode = (
            "int4_awq_clip" if config["algorithm"] == NVModelOptQuantization.Algorithm.AWQ else "int4_rtn_dq"
        )
        q_model = quantize_int4(quantize_mode, model.load_model(), calib_dataloader)

        # save the model to the output path and return the model
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        return model_proto_to_olive_model(q_model, output_model_path, config)
