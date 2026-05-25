# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import logging
import shutil
from pathlib import Path

from olive.common.config_utils import ParamCategory
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, QairtModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.qairt.utils import QairtLogLevel

logger = logging.getLogger(__name__)


class QairtPipelinePass(Pass):
    """Run a QairtPipeline from a YAML recipe on a HuggingFace model.

    Executes the full LLMPipeline workflow (model loading, quantization, compilation)
    defined by the recipe and exports the result as a QairtModelHandler. This pass
    is intended to replace the QairtPreparation -> QairtGenAIBuilder workflow.

    The input HfModelHandler is the authoritative source for the model identity.
    If the recipe also specifies model_id_or_path and it differs from the handler's
    path, an error is raised. If the recipe omits model_id_or_path, the handler's
    path is used.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "recipe": PassConfigParam(
                type_=str,
                required=True,
                category=ParamCategory.PATH,
                description="Path to the YAML recipe file that defines the LLM pipeline stages "
                "(model loading, quantization, genai_builder, etc.).",
            ),
            "cache_dir": PassConfigParam(
                type_=str,
                required=False,
                default_value=None,
                description="Directory for pipeline intermediate artifacts. "
                "Overrides the recipe's cache_dir field when set.",
            ),
            "log_level": PassConfigParam(
                type_=QairtLogLevel,
                required=False,
                default_value=None,
                description="Log level for underlying QAIRT pipeline components. "
                "Valid values: DEBUG, INFO, WARNING, ERROR, TRACE. "
                "Overrides the recipe's log_level field when set.",
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        # Only validates the top-level qairt import. The qairt.experimental.pipeline.*
        # sub-modules are not checked here; if they are absent (e.g. older SDK), the
        # error surfaces in _run_for_config instead.
        try:
            import qairt  # noqa: F401  # pylint: disable=unused-import
        except ImportError as exc:
            raise ImportError(
                "Failed to import QAIRT SDK - please install olive-ai[qairt] to use QAIRT passes. "
                "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
            ) from exc

        return True

    def _run_for_config(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtModelHandler:
        try:
            import qairt  # noqa: F401  # pylint: disable=unused-import
            from qairt.experimental.pipeline.torch.common.recipe import Recipe
            from qairt.experimental.pipeline.torch.llm.pipeline import LLMPipeline
        except ImportError as exc:
            raise ImportError(
                "Failed to import QAIRT Pipeline API - please install olive-ai[qairt] to use QAIRT passes. "
                "If already installed, please run `qairt-vm -i` for help troubleshooting issues."
            ) from exc

        if not isinstance(model, HfModelHandler):
            raise ValueError(f"QairtPipelinePass requires HfModelHandler as input, got {type(model).__name__}")

        recipe_path = Path(config.recipe).resolve()
        if not recipe_path.exists():
            raise ValueError(f"Recipe file not found at: {recipe_path}")

        recipe_data = dict(Recipe.from_file(recipe_path))

        recipe_model_id = recipe_data.get("model_id_or_path")
        if recipe_model_id and recipe_model_id != model.model_path:
            raise ValueError(
                f"Conflict between recipe model_id_or_path '{recipe_model_id}' and input model "
                f"path '{model.model_path}'. Remove model_id_or_path from the recipe or ensure "
                "it matches the input model path."
            )

        if config.cache_dir is not None:
            recipe_data["cache_dir"] = config.cache_dir
        if config.log_level is not None:
            recipe_data["log_level"] = config.log_level

        pipe = LLMPipeline.from_pretrained(model.model_path, recipe=recipe_data)
        pipe.construct()

        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        pipe.export(output_model_path)

        # QairtEncapsulation needs config.json and generation_config.json to generate
        # genai_config.json. Resolve the local HF cache path (model.model_path may be a
        # HuggingFace repo ID rather than a local directory) and copy if not already present.
        try:
            from huggingface_hub import snapshot_download

            local_model_path = snapshot_download(
                model.model_path,
                local_files_only=True,
                ignore_patterns=["*.pt", "*.bin", "*.safetensors"],
            )
        except Exception as e:
            logger.warning(
                "Failed to resolve local HF cache for '%s': %s. File copy will be skipped.",
                model.model_path,
                e,
            )
            local_model_path = model.model_path

        for fname in ("config.json", "generation_config.json"):
            src = Path(local_model_path) / fname
            dst = Path(output_model_path) / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        # The pipeline exports chat_template files into a chat_template/ subdirectory.
        # QairtEncapsulation expects these as flat files in the model root.
        chat_template_dir = Path(output_model_path) / "chat_template"
        for fname in ("chat_template.jinja", "tokenizer_config.json"):
            src = chat_template_dir / fname
            dst = Path(output_model_path) / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        return QairtModelHandler(model_path=output_model_path)
