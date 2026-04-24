# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build ONNX models from HuggingFace model IDs using the mobius package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from olive.common.utils import StrEnumBase
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

    Requires ``mobius-ai`` to be installed::

        pip install mobius-ai

    See https://github.com/microsoft/mobius
    """

    class MobiusRuntime(StrEnumBase):
        """Target runtimes for genai config generation."""

        NONE = "none"
        ORT_GENAI = "ort-genai"

    class MobiusEP(StrEnumBase):
        """Execution providers supported by mobius."""

        DEFAULT = "default"
        CPU = "cpu"
        CUDA = "cuda"
        DML = "dml"
        WEBGPU = "webgpu"
        TRT_RTX = "trt-rtx"
        ONNX_STANDARD = "onnx-standard"

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
                type_=MobiusModelBuilder.MobiusEP,
                required=False,
                default_value=None,
                description=(
                    "Override the mobius execution provider. "
                    "When None (default), the EP is auto-detected from the Olive "
                    "accelerator spec."
                ),
            ),
            "runtime": PassConfigParam(
                type_=MobiusModelBuilder.MobiusRuntime,
                required=False,
                default_value=MobiusModelBuilder.MobiusRuntime.ORT_GENAI,
                description=(
                    "Target runtime. 'ort-genai' (default) generates "
                    "genai_config.json, tokenizer files, and processor "
                    "configs alongside the ONNX models. 'none' to skip."
                ),
            ),
        }

    def _run_for_config(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler | CompositeModelHandler:
        try:
            from mobius import build
        except ImportError as exc:
            raise ImportError(
                "mobius-ai is required to run MobiusModelBuilder. Install with: pip install mobius-ai"
            ) from exc

        if not isinstance(model, HfModelHandler):
            raise ValueError(f"MobiusModelBuilder requires an HfModelHandler input, got {type(model).__name__}.")

        # Resolve EP: explicit config override > accelerator spec > fallback to cpu.
        ep_str: str = config.execution_provider or self.EP_MAP.get(self.accelerator_spec.execution_provider, "cpu")

        dtype_str: str = _PRECISION_TO_DTYPE.get(config.precision, "f32")
        model_id: str = model.model_name_or_path

        # Read trust_remote_code from the model's HuggingFace load kwargs.
        trust_remote_code: bool = model.get_load_kwargs().get("trust_remote_code", False)

        logger.info(
            "MobiusModelBuilder: building '%s' (ep=%s, dtype=%s)",
            model_id,
            ep_str,
            dtype_str,
        )

        if trust_remote_code:
            logger.warning("MobiusModelBuilder: trust_remote_code=True — only use with trusted model sources.")

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        pkg = build(
            model_id,
            dtype=dtype_str,
            execution_provider=ep_str,
            load_weights=True,
            trust_remote_code=trust_remote_code,
        )

        # ModelPackage.save() handles both single and multi-component layouts:
        #   single component  → <output_dir>/model.onnx
        #   multi-component   → <output_dir>/<name>/model.onnx  for each key
        pkg.save(str(output_dir))

        # Generate ORT GenAI config artifacts (genai_config.json, tokenizer
        # files, processor configs) when runtime is set to ort-genai.
        if config.runtime == self.MobiusRuntime.ORT_GENAI:
            self._write_genai_config(pkg, str(output_dir), model_id, ep_str)

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
            additional_files = sorted(
                {str(fp) for fp in output_dir.iterdir()} - {str(onnx_path), str(onnx_path) + ".data"}
            )
            return ONNXModelHandler(
                model_path=str(output_dir),
                onnx_file_name="model.onnx",
                model_attributes={
                    "mobius_package_keys": package_keys,
                    "additional_files": additional_files,
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
            additional_files = sorted(
                {str(fp) for fp in component_dir.iterdir()} - {str(onnx_path), str(onnx_path) + ".data"}
            )
            components.append(
                ONNXModelHandler(
                    model_path=str(component_dir),
                    onnx_file_name="model.onnx",
                    model_attributes={
                        "mobius_component": key,
                        "additional_files": additional_files,
                        **(model.model_attributes or {}),
                    },
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

    @staticmethod
    def _write_genai_config(pkg, output_dir: str, model_id: str, ep: str) -> None:
        """Generate ORT GenAI config artifacts alongside the ONNX models."""
        from mobius.integrations.ort_genai import write_ort_genai_config

        genai_artifacts = write_ort_genai_config(
            pkg, output_dir, hf_model_id=model_id, ep=ep,
        )
        logger.info(
            "MobiusModelBuilder: wrote ORT GenAI config: %s",
            list(genai_artifacts.keys()),
        )
