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
from typing import Optional

from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _get_sd_registry():
    """Import registry from npu_model_gen to keep model_type choices in sync."""
    from model_generate import _SD_CONFIG_REGISTRY
    return _SD_CONFIG_REGISTRY


def _build_fixed_shapes(dim_param: Optional[list], dim_value: Optional[list]) -> Optional[list[str]]:
    """Build --fixed-shapes style list (e.g. ['batch=1', 'height=64']) from dim_param and dim_value."""
    if not dim_param or not dim_value:
        return None
    if len(dim_param) != len(dim_value):
        raise ValueError("dim_param and dim_value must have the same length.")
    return [f"{p}={v}" for p, v in zip(dim_param, dim_value)]


class VitisGenerateModelSD(Pass):
    """Generate Vitis NPU-ready SD submodel (unet or vae_decoder) from ONNX input.
    Use OnnxConversion (PyTorchModel + olive user_script) upstream to produce ONNX.
    Optional dim_param / dim_value override the default fixed shapes used in preprocess (like DynamicToFixedShape).
    """

    @classmethod
    def _default_config(cls, accelerator_spec):
        registry = _get_sd_registry()
        return {
            "model_type": PassConfigParam(
                type_=str,
                required=True,
                description="SD submodel type, must be a key from SD config registry (e.g. sd_unet, sd_vae_decoder, sd_vae_encoder).",
            ),
            "fixed_shapes_dim_param": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description=(
                    "Symbolic dimension names for fixed shapes (e.g. ['batch','channels','height','width']). "
                ),
            ),
            "fixed_shapes_dim_value": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description=(
                    "Defines the values for dimensions listed in fixed_shapes_dim_param (e.g., [1, 4, 64, 64]). "
                    "Use 'x' to preserve a dynamic dimension (e.g., [1, 4, 'x', 'x']). "
                    "The length must match fixed_shapes_dim_param if specified."
                ),
            ),
        }

    @staticmethod
    def _validate_model_type(model_type: str) -> None:
        registry = _get_sd_registry()
        if model_type not in registry:
            raise ValueError(
                f"model_type must be one of {list(registry.keys())}, got {model_type!r}"
            )

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: BasePassConfig,
        output_model_path: str,
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise TypeError(
                "VitisGenerateModelSD requires ONNXModelHandler (run OnnxConversion first). "
                f"Got {type(model).__name__}"
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

        fixed_shapes = _build_fixed_shapes(
            getattr(config, "fixed_shapes_dim_param", None), getattr(config, "fixed_shapes_dim_value", None)
        )
        if fixed_shapes:
            logger.info(
                "[VitisGenerateModelSD] Overriding fixed shapes: %s",
                fixed_shapes,
            )

        from model_generate import generate_sd_model

        generate_sd_model(
            input_model=str(onnx_input_path),
            output_dir=str(output_dir),
            model_type=model_type,
            fixed_shapes=fixed_shapes,
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
