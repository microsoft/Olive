# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import Union

from olive.common.utils import hardlink_copy_file
from olive.hardware import AcceleratorSpec
from olive.model import HfModelHandler, QairtModelHandler, QairtPreparedModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

import qairt
import qairt.gen_ai_api as qairt_genai

logger = logging.getLogger(__name__)


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
                type_=str,
                default_value=None,
                description="Log level to be used within underlying QAIRT components."
                "Valid values: DEBUG, INFO, WARN, ERROR.",
            ),
            # Device configs
            "backend": PassConfigParam(
                type_=str,
                default_value="CPU",
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
            # Model configs
            "multi_graph": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Produces context binaries with additional context length combinations. "
                "Improves token generation performance for different context lengths but increases preparation time. "
                "HTP only."
            ),
            "num_splits": PassConfigParam(
                type_=int,
                default_value=-1,
                description="Number of model splits. Default value is -1 (value chosen by QAIRT based on model). " 
                "HTP only.",
            ),
        }

    
    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        
        if config.backend != qairt.BackendType.HTP.value:
            if config.multi_graph:
                logger.error("multi_graph is unsupported on non-HTP backends")
                return False
            if config.num_splits != -1:
                logger.error("num_splits is unsupported on non-HTP backends")
                return False
        
        return True
        

    def _run_for_config(
        self,
        model: Union[HfModelHandler, QairtPreparedModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtModelHandler:

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

        gen_ai_builder = qairt_genai.GenAIBuilderFactory.create(
            pretrained_model_path=Path(model.model_path) / "base" / "onnx",
            backend_type=config.backend,
            cache_root=config.cache_dir,
            tokenizer_path=Path(model.model_path),
            config_path=Path(model.model_path)
        )

        # Embedding LUT is unsupported for now
        gen_ai_builder._prepare_embedding_lut = False

        # Can only set target and transformation configurations if the BE is HTP
        if config.backend == qairt.BackendType.HTP.value:
            gen_ai_builder.set_targets([config.soc_details])
            gen_ai_builder.multi_graph = config.multi_graph
            if config.num_splits != -1:
                gen_ai_builder._transformation_config.model_transformer_config.split_model.num_splits = config.num_splits

        gen_ai_container = gen_ai_builder.build()

        # Handling of UDMA on LLMContainer
        #if config.backend == qairt.BackendType.HTP.value:
            # TODO fix index issue here where index should map to index
            #htp_version = gen_ai_container._backend_extensions_config.device_custom_configs[0].dsp_arch
            #if htp_version >= "v81":
                #gen_ai_builder._backend_extensions_config.context_custom_configs[0].extended_udma = True

        gen_ai_container.save(output_model_path, exist_ok=True)

        # QairtModelHandler requires certain source model files to be passed through
        passthrough_files = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        for file in passthrough_files:
            config_path = Path(model.model_path) / file
            dest_path = Path(output_model_path)
            # TODO Remove once we have NB1 scripts for all models
            try:
                hardlink_copy_file(config_path, dest_path, follow_symlinks=True)
            except:
                pass

        return QairtModelHandler(model_path=output_model_path)
