# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import subprocess
import sys
from pathlib import Path

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class ConvertHfToGGUF(Pass):
    """Convert the test HuggingFace model directory to a GGUF file."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "llama_cpp_env_path": PassConfigParam(
                type_=str,
                default_value="llama_env",
                description="Path to the llama.cpp virtual environment containing convert_hf_to_gguf.py.",
            ),
            "reference_model_path": PassConfigParam(
                type_=str,
                default_value=None,
                description="Fallback model path to convert when test_model_path is not set.",
            ),
            "gguf_file_name": PassConfigParam(
                type_=str,
                default_value="model.gguf",
                description="GGUF output filename.",
            ),
        }

    @staticmethod
    def _get_python_executable(env_path: Path) -> str:
        if sys.platform.startswith("win"):
            return str(env_path / "Scripts" / "python.exe")
        return str(env_path / "bin" / "python")

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        source = model.test_model_path or config.reference_model_path
        if not source:
            logger.info("ConvertHfToGGUF skipped: no source model directory was provided.")
            return model

        source_path = Path(source)
        if not source_path.is_dir():
            logger.info("ConvertHfToGGUF skipped: source model directory does not exist: %s", source_path)
            return model

        gguf_path = source_path / config.gguf_file_name
        if gguf_path.exists():
            logger.info("ConvertHfToGGUF skipped: GGUF already exists at %s", gguf_path)
            model_attributes = dict(model.model_attributes) if model.model_attributes else {}
            model_attributes["reference_gguf_model_path"] = str(gguf_path)
            model.model_attributes = model_attributes
            return model

        env_path = Path(config.llama_cpp_env_path).resolve()
        convert_script = env_path / "convert_hf_to_gguf.py"
        conversion_pkg = env_path / "conversion"
        python_path = self._get_python_executable(env_path)

        if not Path(python_path).exists():
            raise RuntimeError(f"Could not find llama_env python executable: {python_path}")
        if not convert_script.exists():
            raise RuntimeError(f"Could not find convert_hf_to_gguf.py at: {convert_script}")
        if not conversion_pkg.exists():
            raise RuntimeError(f"Could not find conversion package at: {conversion_pkg}")

        subprocess.run(
            [python_path, str(convert_script), str(source_path), "--outfile", str(gguf_path), "--outtype", "f32"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Converted test model to GGUF at %s", gguf_path)

        model_attributes = dict(model.model_attributes) if model.model_attributes else {}
        model_attributes["reference_gguf_model_path"] = str(gguf_path)
        model.model_attributes = model_attributes
        return model
