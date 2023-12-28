# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from olive.cache import get_local_path_from_root
from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import OliveModelHandler
from olive.model.handler import OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

if TYPE_CHECKING:
    from openvino import CompiledModel


logger = logging.getLogger(__name__)


def _default_validate_func(model: "CompiledModel", validation_loader) -> float:
    import numpy as np
    from sklearn.metrics import accuracy_score

    predictions = []
    references = []

    output = model.outputs[0]

    for data_item, target in validation_loader:
        pred = model(data_item)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


class ModelTypeEnum(str, Enum):
    TRANSFORMER = "TRANSFORMER"


class PresetEnum(str, Enum):
    PERFORMANCE = "PERFORMANCE"
    MIXED = "MIXED"


class IgnoreScopeTypeEnum(str, Enum):
    NAMES = "names"
    TYPES = "types"
    PATTERNS = "patterns"


class DropTypeEnum(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    RELATIVE = "RELATIVE"


class OpenVINOQuantizationBase(Pass):
    """Post-training quantization for OpenVINO model.

    Please refer to https://docs.openvino.ai/2023.2/ptq_introduction.html for more details.
    """

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                category=ParamCategory.OBJECT,
                description=(
                    "Function/function name to generate dataloader for calibration, required if data_config is None."
                ),
            ),
            "dataloader_func_kwargs": PassConfigParam(
                type_=Dict[str, Any],
                description="Keyword arguments for dataloader_func.",
            ),
            "data_dir": PassConfigParam(
                type_=OLIVE_RESOURCE_ANNOTATIONS,
                category=ParamCategory.DATA,
                description=(
                    "Path to the directory containing the dataset. For local data, it is required if dataloader_func"
                    " is provided."
                ),
            ),
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="Data config for calibration, required if dataloader_func is None.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                description="Data config for calibration, required if dataloader_func is None.",
            ),
            "model_type": PassConfigParam(
                type_=ModelTypeEnum,
                required=False,
                default_value=None,
                description=(
                    "Used to specify quantization scheme required for specific type of the model. "
                    "'TRANSFORMER' is the only supported special quantization scheme to preserve accuracy "
                    "after quantization of Transformer models (BERT, DistilBERT, etc.). None is default."
                ),
            ),
            "preset": PassConfigParam(
                type_=PresetEnum,
                required=False,
                default_value=PresetEnum.PERFORMANCE,
                description=("Defines quantization scheme for the model. Supported values: 'PERFORMANCE', 'MIXED'."),
            ),
            "ignored_scope": PassConfigParam(
                type_=Union[str, List[str]],
                required=False,
                default_value=None,
                description=(
                    "This parameter can be used to exclude some layers "
                    "from the quantization process to preserve the model accuracy. Please refer to "
                    "https://docs.openvino.ai/2023.2/basic_quantization_flow.html#tune-quantization-parameters."
                ),
            ),
            "ignored_scope_type": PassConfigParam(
                type_=IgnoreScopeTypeEnum,
                required=False,
                default_value=None,
                description=("Defines the type of the ignored scope. Supported values: 'names', 'types', 'patterns'."),
            ),
            "target_device": PassConfigParam(
                type_=Device,
                required=False,
                default_value=accelerator_spec.accelerator_type,
                description=(
                    "Target device for the model. "
                    "Supported values: 'any', 'cpu', 'gpu', 'cpu_spr', 'vpu'. "
                    "Default value is the same as the accelerator type of this workflow run."
                ),
            ),
            "extra_configs": PassConfigParam(
                type_=List[Dict],
                required=False,
                description=(
                    "Extra configurations for OpenVINO model quantization. Please refer to "
                    "https://docs.openvino.ai/2023.2/basic_quantization_flow.html#tune-quantization-parameters."
                ),
            ),
        }

    @staticmethod
    def _create_calibration_dataset(common_dataloader):
        """Create an nncf.Dataset instance from a common dataloader."""
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO pass") from None

        def transform_fn(data_item):
            data, _ = data_item
            return data

        return nncf.Dataset(common_dataloader, transform_fn)

    def _get_nncf_dataset(self, config, data_root):
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO pass") from None

        if config["dataloader_func"]:
            data_dir = get_local_path_from_root(data_root, config["data_dir"])
            data_loader = self._user_module_loader.call_object(
                config["dataloader_func"], data_dir, config["batch_size"], **(config["dataloader_func_kwargs"] or {})
            )
        elif config["data_config"]:
            data_config = validate_config(config["data_config"], DataConfig)
            data_loader = data_config.to_data_container().create_dataloader(data_root)

        def transform_fn(data_item):
            data, _ = data_item
            return data

        return nncf.Dataset(data_loader, transform_fn)

    @staticmethod
    def _get_extra_params(config):
        import nncf

        device_map = {
            "cpu": nncf.TargetDevice.CPU,
            "gpu": nncf.TargetDevice.CPU,
            "cpu_spr": nncf.TargetDevice.CPU_SPR,
            "vpu": nncf.TargetDevice.VPU,
            "npu": nncf.TargetDevice.VPU,
        }

        extra_params = {}
        extra_params["model_type"] = nncf.ModelType.Transformer if config.get("model_type") == "TRANSFORMER" else None
        extra_params["preset"] = (
            nncf.QuantizationPreset.PERFORMANCE
            if config.get("preset") == "PERFORMANCE"
            else nncf.QuantizationPreset.MIXED
        )
        extra_params["target_device"] = device_map.get(config.get("target_device"), nncf.TargetDevice.ANY)

        if config.get("ignored_scope"):
            kwargs = {config.get("ignored_scope_type"): config.get("ignored_scope")}
            extra_params["ignored_scopes"] = nncf.IgnoredScope(**kwargs)

        return extra_params


class OpenVINOQuantization(OpenVINOQuantizationBase):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return OpenVINOQuantizationBase._default_config(accelerator_spec)

    def _run_for_config(
        self, model: OpenVINOModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> OpenVINOModelHandler:
        try:
            import nncf
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        assert config["dataloader_func"] or config["data_config"], "dataloader_func or data_config is required."

        calibration_dataset = self._get_nncf_dataset(config, data_root)
        model = model.load_model()
        extra_params = self._get_extra_params(config)

        quantized_model = nncf.quantize(model, calibration_dataset, **extra_params)

        model_name = "ov_model"
        output_dir = Path(output_model_path) / model_name
        ov.save_model(quantized_model, output_model=output_dir.with_suffix(".xml"))
        return OpenVINOModelHandler(model_path=output_model_path)


class OpenVINOQuantizationWithAccuracy(OpenVINOQuantizationBase):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "validation_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                category=ParamCategory.OBJECT,
                description=(
                    "Used to compute accuracy metric. "
                    "Validation function receives openvino.runtime.CompiledModel object "
                    "and validation dataloader and returns accuracy metric value."
                ),
            ),
            "max_drop": PassConfigParam(
                type_=float,
                default_value=0.01,
                description=(
                    "Defines the accuracy drop threshold. The quantization process stops "
                    "when the degradation of accuracy metric on the validation dataset is less than the max_drop. "
                    "NNCF will stop the quantization and report an error if the max_drop value can't be reached. "
                    "The default value is 0.01."
                ),
            ),
            "drop_type": PassConfigParam(
                type_=DropTypeEnum,
                required=False,
                default_value=DropTypeEnum.ABSOLUTE,
                description=(
                    "Defines the type of the max_drop. Supported values: 'ABSOLUTE', 'RELATIVE'. "
                    "The default value is 'ABSOLUTE'."
                ),
            ),
        }
        config.update(OpenVINOQuantizationBase._default_config(accelerator_spec))
        return config

    def _run_for_config(
        self, model: OliveModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:
        try:
            import nncf
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        assert config["dataloader_func"] or config["data_config"], "dataloader_func or data_config is required."

        calibration_dataset = self._get_nncf_dataset(config, data_root)
        validation_dataset = self._get_nncf_dataset(config, data_root)

        model = model.load_model()
        extra_params = self._get_extra_params(config)

        validate_func = (
            self._user_module_loader.load_object(config["validation_func"])
            if config.get("validation_func")
            else _default_validate_func
        )

        drop_type = nncf.DropType.ABSOLUTE if config["drop_type"] == "ABSOLUTE" else nncf.DropType.RELATIVE

        quantized_model = nncf.quantize_with_accuracy_control(
            model,
            calibration_dataset=calibration_dataset,
            validation_dataset=validation_dataset,
            validation_fn=validate_func,
            max_drop=config["max_drop"],
            drop_type=drop_type,
            **extra_params
        )

        model_name = "ov_model"
        output_dir = Path(output_model_path) / model_name
        ov.save_model(quantized_model, output_model=output_dir.with_suffix(".xml"))
        return OpenVINOModelHandler(model_path=output_model_path)
