# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import Union

from packaging.version import Version

from olive.common.utils import StrEnumBase, hardlink_copy_file
from olive.hardware import AcceleratorSpec
from olive.model import HfModelHandler, QairtModelHandler, QairtPreparedModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.qairt.utils import QairtLogLevel

logger = logging.getLogger(__name__)


class QairtBackend(StrEnumBase):
    CPU = "CPU"
    HTP = "HTP"


class QairtGenAIBuilder(Pass):
    """Create a QairtModelHandler from a QairtPreparedModelHandler.

    Applies various QAIRT-specific optimizations depending on model architecture,
    converts them to DLC, and compiles a context binary compatible with the specified SoC.
    Uses QAIRT GenAIBuilder Python API from the QAIRT SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            # General configs
            "cache_dir": PassConfigParam(
                type_=str,
                default_value="./cache/qairt/gen_ai_builder",
                description="Directory to be used as the cache directory for subsequent GenAIBuilder invocations."
                "By default, saves to a similar location to the Olive cache.",
            ),
            "log_level": PassConfigParam(
                type_=QairtLogLevel,
                default_value=None,
                description="Log level to be used within underlying QAIRT components."
                "Valid values: DEBUG, INFO, WARN, ERROR.",
            ),
            # Device configs
            "backend": PassConfigParam(
                type_=QairtBackend,
                default_value=QairtBackend.CPU,
                description="Target accelerator to prepare the model for. Accepted values are 'CPU' and 'HTP'.",
            ),
            "soc_details": PassConfigParam(
                type_=str,
                default_value=None,
                description="Device specification to use for compilation. Can be specified"
                " as a spec string in the form 'chipset:value;dsp_arch:value;soc_model:value|...'."
                " This option will be ignored if any device custom configurations are set."
                "e.g. 'chipset:chipset:SC8380XP', 'dsp_arch:v73;soc_model:60' ",
            ),
            "vtcm_size_in_mb": PassConfigParam(
                type_=int, default_value=0, description="VTCM size in megabytes for HTP execution. HTP only."
            ),
            "hvx_threads": PassConfigParam(
                type_=int, default_value=0, description="Number of HVX threads for parallel processing. HTP only."
            ),
            "extended_udma": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Improves performance at the cost of memory by using UDMA. HTP only.",
            ),
            # Model configs
            "sequence_lengths": PassConfigParam(
                type_=list[int], default_value=None, description="The sequence lengths of the final compiled graphs."
            ),
            "native_kv": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Utilizes native buffers for KVCache updates. Only compatible with sequence_lengths: [32, 128]. HTP only.",
            ),
            "num_splits": PassConfigParam(
                type_=int,
                default_value=-1,
                description="Number of model splits. Default value is -1 (value chosen by QAIRT based on model). "
                "HTP only.",
            ),
            "multi_graph": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Produces context binaries with additional context length combinations. "
                "Improves token generation performance for different context lengths but increases preparation time. "
                "HTP only.",
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        try:
            import qairt
        except ImportError as exc:
            raise ImportError(
                "Failed to import QAIRT GenAIBuilder API - please install olive-ai[qairt] to use QAIRT passes."
                "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
            ) from exc

        if config.backend != qairt.BackendType.HTP.value:
            if config.extended_udma:
                logger.error("extended_udma is unsupported on non-HTP backends")
                return False
            if config.vtcm_size_in_mb != 0:
                logger.error("vtcm_size_in_mb is unsupported on non-HTP backends")
                return False
            if config.hvx_threads != 0:
                logger.error("hvx_threads is unsupported on non-HTP backends")
                return False
            if config.sequence_lengths:
                logger.error("sequence_lengths is unsupported on non-HTP backends")
                return False
            if config.native_kv:
                logger.error("native_kv is unsupported on non-HTP backends")
                return False
            if config.num_splits != -1:
                logger.error("num_splits is unsupported on non-HTP backends")
                return False
            if config.multi_graph:
                logger.error("multi_graph is unsupported on non-HTP backends")
                return False

        native_kv_supported_sequence_lengths = [[32, 128]]
        if config.native_kv and config.sequence_lengths not in native_kv_supported_sequence_lengths:
            logger.error(
                "native_kv is only supported for the following sequence lengths: %s",
                native_kv_supported_sequence_lengths,
            )
            return False

        return True

    def _run_for_config(
        self,
        model: Union[HfModelHandler, QairtPreparedModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtModelHandler:
        try:
            import qairt
            import qairt.gen_ai_api as qairt_genai
        except ImportError as exc:
            raise ImportError(
                "Failed to import QAIRT GenAIBuilder API - please install olive-ai[qairt] to use QAIRT passes."
                "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
            ) from exc

        from qairt import __sdk_version__ as sdk_version

        if Version(sdk_version) < Version("2.45.0"):
            raise OSError("QairtGenAIBuilder pass is unsupported for QAIRT versions < 2.45.0")

        if config.log_level:
            os.environ["QAIRT_LOG_LEVEL"] = config.log_level

        if not config.cache_dir:
            logger.warning(
                "QAIRT GenAIBuilder cache directory not set. Using this will decrease future preparation time."
            )

        if config.backend == qairt.BackendType.CPU.value and not isinstance(model, HfModelHandler):
            raise ValueError("QAIRT CPU GenAIBuilder can only consume HfModelHandler")

        if config.backend == qairt.BackendType.HTP.value and not isinstance(model, QairtPreparedModelHandler):
            raise ValueError("QAIRT HTP GenAIBuilder can only consume QairtPreparedModelHandler")

        if isinstance(model, QairtPreparedModelHandler):
            pretrained_model_path = Path(model.model_path) / "base" / "onnx"
        else:
            pretrained_model_path = Path(model.model_path)

        gen_ai_builder = qairt_genai.GenAIBuilderFactory.create(
            pretrained_model_path=pretrained_model_path,
            backend_type=config.backend,
            cache_root=config.cache_dir,
            tokenizer_path=Path(model.model_path),
            config_path=Path(model.model_path),
        )

        # QAIRT 2.45.0 requires the following environment variable for advanced functionality
        if sdk_version == "2.45.0":
            os.environ["QAIRT_USE_NEW_ARCL_ALGO"] = os.getenv("QAIRT_USE_NEW_ARCL_ALGO", "1")

        # pylint: disable=protected-access
        # QairtGenAIBuilder requires direct access to these configuration attributes

        # Embedding LUT is unsupported for now
        gen_ai_builder._prepare_embedding_lut = False

        # Can only set target and transformation configurations if the BE is HTP
        if config.backend == qairt.BackendType.HTP.value:
            # Device configs
            if config.soc_details:
                gen_ai_builder.set_targets([config.soc_details])

            if config.vtcm_size_in_mb != 0:
                gen_ai_builder._compilation_config.graph_custom_configs[0].vtcm_size_in_mb = config.vtcm_size_in_mb

            if config.hvx_threads != 0:
                gen_ai_builder._compilation_config.graph_custom_configs[0].hvx_threads = config.hvx_threads

            if config.extended_udma:
                dev_cfg = gen_ai_builder._compilation_config.device_custom_configs[0]
                arch_version = int(str(dev_cfg.dsp_arch).lstrip("v"))
                if arch_version >= 81:
                    gen_ai_builder._compilation_config.context_custom_configs[0].extended_udma = True
                else:
                    raise ValueError("extended_udma is unsupported on DSP architectures less than v81")

            # Model configs
            if config.sequence_lengths:
                gen_ai_builder._transformation_config.model_transformer_config.arn_cl_options.auto_regression_number = (
                    config.sequence_lengths
                )

            # NativeKV should be enabled after sequence lengths are modified
            if config.native_kv:
                gen_ai_builder.native_kv = config.native_kv

            if config.num_splits != -1:
                gen_ai_builder._transformation_config.model_transformer_config.split_model.num_splits = (
                    config.num_splits
                )

            gen_ai_builder.multi_graph = config.multi_graph

        gen_ai_container = gen_ai_builder.build()
        gen_ai_container.save(output_model_path, exist_ok=True)

        # QairtModelHandler requires certain source model files to be passed through
        passthrough_files = [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        for file in passthrough_files:
            config_path = Path(model.model_path) / file
            dest_path = Path(output_model_path)
            try:
                hardlink_copy_file(config_path, dest_path, follow_symlinks=True)
            except (FileNotFoundError, OSError, ValueError):
                # Not every model has all the files listed above
                pass

        model_attrs = {"sequence_lengths": config.sequence_lengths} if config.sequence_lengths else {}
        return QairtModelHandler(model_path=output_model_path, model_attributes=model_attrs)
