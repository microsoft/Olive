# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------

"""Olive Pass for Vitis NPU Stable Diffusion submodel generation.

Accepts ONNX input only; run OnnxConversion to produce ONNX input model first,
then this pass runs generate_sd_model to generate NPU-ready models.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class VitisGenerateModelSD(Pass):
    """Generate Vitis NPU-ready SD submodel from ONNX input.

    Use OnnxConversion to produce ONNX input model.
    Optional resolutions to generate NPU-ready models. Default is [512x512].
    """

    @classmethod
    def _default_config(cls, accelerator_spec):
        return {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description="SD submodel type.",
            ),
            "resolutions": PassConfigParam(
                type_=list[str],
                default_value=["512x512"],
                required=False,
                description="List of resolutions (e.g. ['512x512', '1024x1024']) Default is [512x512].",
            ),
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: BasePassConfig,
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            from model_generate import generate_model
        except ImportError as e:
            raise ImportError(
                "model_generate is required for VitisGenerateModelSD. Please install the model_generate package."
            ) from e

        if not isinstance(model, ONNXModelHandler):
            raise TypeError(
                f"VitisGenerateModelSD requires ONNXModelHandler (run OnnxConversion first). Got {type(model).__name__}"
            )
        model_type = config.model_type

        output_dir = Path(output_model_path)
        if output_dir.suffix == ".onnx":
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[VitisGenerateModelSD] output_dir=%s, model_type=%s",
            output_dir,
            model_type,
        )

        onnx_input_path = self.resolve_onnx_input_path(model)
        logger.info("[VitisGenerateModelSD] ONNX input path: %s", onnx_input_path)

        resolutions = getattr(config, "resolutions", None)
        extra_options = {"model_type": model_type}
        if resolutions:
            logger.info(
                "[VitisGenerateModelSD] Using resolutions: %s",
                resolutions,
            )
            extra_options["resolutions"] = ",".join(resolutions)

        generate_model(
            mode="sd",
            input_model=str(onnx_input_path),
            output_dir=str(output_dir),
            extra_options=extra_options,
        )

        self._ensure_model_onnx(output_dir)

        return ONNXModelHandler(
            model_path=str(output_dir),
            onnx_file_name="model.onnx",
        )

    def resolve_onnx_input_path(self, model: ONNXModelHandler) -> Path:
        p = Path(model.model_path)
        if p.is_file():
            return p
        if p.is_dir():
            name = getattr(model, "onnx_file_name", None)
            if name:
                f = p / name
                if f.exists():
                    return f
                raise FileNotFoundError(f"Specified onnx_file_name does not exist under {p}: {name}")

            default_model_path = p / "model.onnx"
            if default_model_path.exists():
                return default_model_path

            onnx_files = sorted(path for path in p.glob("*.onnx") if path.is_file())
            if len(onnx_files) == 1:
                return onnx_files[0]
            if len(onnx_files) > 1:
                candidates = ", ".join(path.name for path in onnx_files)
                raise ValueError(
                    f"Multiple .onnx model files found under {p}: {candidates}. Please specify one using the onnx_file_name argument."
                )
            else:
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
            raise FileNotFoundError(
                f"[VitisGenerateModelSD] No optimized.onnx or dd/replaced.onnx found under {output_dir}. Please check the output directory.",
            )
