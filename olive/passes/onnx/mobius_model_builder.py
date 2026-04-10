# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build ONNX models from HuggingFace model IDs using the mobius package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from olive.constants import Precision
from olive.hardware.constants import ExecutionProvider
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.passes.pass_config import BasePassConfig

logger = logging.getLogger(__name__)

# Maps Olive Precision values to mobius dtype strings.
# "f32" = 32-bit float (torch.float32), standard full precision.
# "f16" = 16-bit float (torch.float16), half precision — good for GPU inference.
# "bf16" = bfloat16 (torch.bfloat16), brain float — preferred over f16 on newer hardware.
# For INT4/INT8 quantization, use a downstream Olive quantization pass (e.g. OnnxMatMulNBits)
# after this pass rather than setting precision here.
_PRECISION_TO_DTYPE: dict[str, str] = {
    Precision.FP32: "f32",
    Precision.FP16: "f16",
    Precision.BF16: "bf16",
}


class MobiusModelBuilder(Pass):
    """Olive pass that uses mobius to build ONNX models from HuggingFace model IDs.

    Supports all model architectures registered in mobius (LLMs, VLMs, speech
    models, diffusion models).  For multi-component models (e.g. vision-language
    models that produce ``model``, ``vision``, and ``embedding`` sub-graphs) the
    pass returns a :class:`~olive.model.handler.composite.CompositeModelHandler`
    whose components are individual :class:`~olive.model.ONNXModelHandler` objects.
    Single-component models return a plain :class:`~olive.model.ONNXModelHandler`.

    Requires ``mobius-genai`` to be installed::

        pip install mobius-genai

    See https://github.com/microsoft/mobius
    """

    # Maps Olive ExecutionProvider enum values to mobius EP names.
    EP_MAP: ClassVar[dict[ExecutionProvider, str]] = {
        ExecutionProvider.CPUExecutionProvider: "cpu",
        ExecutionProvider.CUDAExecutionProvider: "cuda",
        ExecutionProvider.DmlExecutionProvider: "dml",
        ExecutionProvider.WebGpuExecutionProvider: "webgpu",
    }

    @classmethod
    def is_accelerator_agnostic(cls, accelerator_spec: AcceleratorSpec) -> bool:
        # EP selection determines which fused ops are emitted, so this pass is
        # EP-specific.
        return False

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=Precision,
                required=False,
                default_value=Precision.FP32,
                description=(
                    "Model weight / compute precision. One of: fp32, fp16, bf16. "
                    "Defaults to fp32. For INT4 quantization, run an Olive "
                    "quantization pass (e.g. OnnxMatMulNBits) after this pass."
                ),
            ),
            "execution_provider": PassConfigParam(
                type_=str,
                required=False,
                default_value=None,
                description=(
                    "Override the mobius execution provider (cpu, cuda, dml, webgpu). "
                    "When None (default), the EP is auto-detected from the Olive "
                    "accelerator spec."
                ),
            ),
            "trust_remote_code": PassConfigParam(
                type_=bool,
                required=False,
                default_value=False,
                description="Pass trust_remote_code=True to the HuggingFace config loader.",
            ),
        }

    def _run_for_config(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler | CompositeModelHandler:
        try:
            from mobius import build  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "mobius-genai is required to run MobiusModelBuilder. Install with: pip install mobius-genai"
            ) from exc

        if not isinstance(model, HfModelHandler):
            raise ValueError(f"MobiusModelBuilder requires an HfModelHandler input, got {type(model).__name__}.")

        # Resolve EP: explicit config override > accelerator spec > fallback to cpu.
        ep_str: str = config.execution_provider or self.EP_MAP.get(self.accelerator_spec.execution_provider, "cpu")

        dtype_str: str = _PRECISION_TO_DTYPE.get(config.precision, "f32")
        model_id: str = model.model_name_or_path

        logger.info(
            "MobiusModelBuilder: building '%s' (ep=%s, dtype=%s)",
            model_id,
            ep_str,
            dtype_str,
        )

        if config.trust_remote_code:
            logger.warning("MobiusModelBuilder: trust_remote_code=True — only use with trusted model sources.")

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        pkg = build(
            model_id,
            dtype=dtype_str,
            execution_provider=ep_str,
            load_weights=True,
            trust_remote_code=config.trust_remote_code,
        )

        # ModelPackage.save() handles both single and multi-component layouts:
        #   single component  → <output_dir>/model.onnx
        #   multi-component   → <output_dir>/<name>/model.onnx  for each key
        pkg.save(str(output_dir))

        package_keys = list(pkg.keys())
        logger.info("MobiusModelBuilder: saved components %s to '%s'", package_keys, output_dir)

        if len(package_keys) == 1:
            # Single-component model (most LLMs): return a plain ONNXModelHandler.
            onnx_path = output_dir / "model.onnx"
            if not onnx_path.exists():
                raise RuntimeError(
                    f"MobiusModelBuilder: expected output file not found: {onnx_path}. "
                    "mobius.build() may have failed silently or saved to an unexpected path."
                )
            return ONNXModelHandler(
                model_path=str(output_dir),
                onnx_file_name="model.onnx",
                model_attributes={
                    "mobius_package_keys": package_keys,
                    **(model.model_attributes or {}),
                },
            )

        # Multi-component model (VLMs, encoder-decoders, diffusion pipelines):
        # mobius saves each component to <output_dir>/<key>/model.onnx.
        components = []
        for key in package_keys:
            component_dir = output_dir / key
            onnx_path = component_dir / "model.onnx"
            if not onnx_path.exists():
                raise RuntimeError(
                    f"MobiusModelBuilder: expected output file not found: {onnx_path}. "
                    f"mobius.build() may have failed silently for component '{key}'."
                )
            components.append(
                ONNXModelHandler(
                    model_path=str(component_dir),
                    onnx_file_name="model.onnx",
                    model_attributes={"mobius_component": key},
                )
            )

        return CompositeModelHandler(
            model_components=components,
            model_component_names=package_keys,
            model_path=str(output_dir),
            model_attributes={
                "mobius_package_keys": package_keys,
                **(model.model_attributes or {}),
            },
        )
