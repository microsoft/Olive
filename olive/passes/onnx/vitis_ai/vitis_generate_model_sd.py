#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Olive Pass for Vitis NPU Stable Diffusion submodel generation (UNet / VAE decoder).

Accepts ONNX input only; run OnnxConversion (e.g. from PyTorchModel + olive user_script) first,
then this pass runs generate_sd_model for preprocess + partition.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _get_sd_registry():
    """Import registry from npu_model_gen to keep model_type choices in sync."""
    import model_generate

    return model_generate.SUPPORTED_SD_MODEL_TYPES


class VitisGenerateModelSD(Pass):
    """Generate Vitis NPU-ready SD submodel (unet or vae_decoder) from ONNX input.

    Use OnnxConversion to produce ONNX input model.
    Optional resolutions to generate NPU-ready models. Default is [512x512].
    """

    @classmethod
    def _default_config(cls, accelerator_spec):
        registry = _get_sd_registry()
        return {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description=f"SD submodel type, must be one of {', '.join(registry)}.",
            ),
            "resolutions": PassConfigParam(
                type_=list[str],
                default_value=["512x512"],
                required=False,
                description="List of resolutions (e.g. ['512x512', '1024x1024']) Default is [512x512].",
            ),
        }

    @staticmethod
    def _validate_model_type(model_type: str) -> None:
        registry = _get_sd_registry()
        if model_type not in registry:
            raise ValueError(f"model_type must be one of {list(registry.keys())}, got {model_type!r}")

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: BasePassConfig,
        output_model_path: str,
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise TypeError(
                f"VitisGenerateModelSD requires ONNXModelHandler (run OnnxConversion first). Got {type(model).__name__}"
            )
        model_type = config.model_type
        self._validate_model_type(model_type)

        output_dir = Path(output_model_path)
        if output_dir.suffix == ".onnx":
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[VitisGenerateModelSD] output_dir=%s, model_type=%s",
            output_dir,
            model_type,
        )

        onnx_input_path = self._resolve_onnx_input_path(model)
        logger.info("[VitisGenerateModelSD] ONNX input path: %s", onnx_input_path)

        resolutions = getattr(config, "resolutions", None)
        if resolutions:
            logger.info(
                "[VitisGenerateModelSD] Using resolutions: %s",
                resolutions,
            )

        from model_generate import generate_sd_model

        generate_sd_model(
            input_model=str(onnx_input_path),
            output_dir=str(output_dir),
            model_type=model_type,
            resolutions=resolutions,
        )

        self._ensure_model_onnx(output_dir)

        return ONNXModelHandler(
            model_path=str(output_dir),
            onnx_file_name="model.onnx",
        )

    def _resolve_onnx_input_path(self, model: ONNXModelHandler) -> Path:
        p = Path(model.model_path)
        if p.is_file():
            return p
        if p.is_dir():
            name = getattr(model, "onnx_file_name", None)
            if name:
                f = p / name
                if f.exists():
                    return f
            onnx_files = list(p.glob("*.onnx"))
            if onnx_files:
                return onnx_files[0]
            raise FileNotFoundError(f"No .onnx file found under {p}")
        raise FileNotFoundError(f"Model path does not exist: {p}")

    def _ensure_model_onnx(self, output_dir: Path) -> None:
        """Copy actual generate_sd_model output to output_dir/model.onnx if needed."""
        model_onnx = output_dir / "model.onnx"
        if model_onnx.exists():
            return
        optimized = output_dir / "optimized.onnx"
        dd_replaced = output_dir / "dd" / "replaced.onnx"
        if dd_replaced.exists():
            shutil.copy2(dd_replaced, model_onnx)
            logger.info("[VitisGenerateModelSD] Wrote model.onnx from dd/replaced.onnx")
        elif optimized.exists():
            shutil.copy2(optimized, model_onnx)
            logger.info("[VitisGenerateModelSD] Wrote model.onnx from optimized.onnx")
        else:
            logger.warning(
                "[VitisGenerateModelSD] No optimized.onnx or dd/replaced.onnx found under %s",
                output_dir,
            )
