# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import json
from pathlib import Path

from olive_mcp.constants import (
    PROVIDER_TO_EXTRAS,
    PROVIDER_TO_GENAI,
    Command,
)

# Maps the MCP "implementation" parameter to the candidate pass types that
# olive/cli/quantize.py would select.  This is *structural* routing info —
# the actual package dependencies come from olive_config.json at runtime.
_IMPL_TO_PASS_TYPES: dict[str, list[str]] = {
    "bnb": ["OnnxBnb4Quantization"],
    "inc": ["IncDynamicQuantization", "IncQuantization", "IncStaticQuantization"],
    "autogptq": ["GptqQuantizer"],
    "awq": ["AutoAWQQuantizer"],
    "olive": [
        "Gptq",
        "Rtn",
        "OnnxBlockWiseRtnQuantization",
        "OnnxHqqQuantization",
    ],
    "ort": ["OnnxDynamicQuantization", "OnnxStaticQuantization"],
    "nvmo": ["NVModelOptQuantization"],
    "aimet": ["AimetQuantization"],
    "quarot": ["QuaRot"],
    "spinquant": ["SpinQuant"],
}


@functools.lru_cache
def _load_olive_config() -> dict | None:
    """Load olive_config.json as raw JSON.

    Tries the installed ``olive`` package first, then falls back to the
    repo-relative path (for development).
    """
    config_path: Path | None = None
    try:
        from olive.package_config import OlivePackageConfig

        config_path = Path(OlivePackageConfig.get_default_config_path())
    except ImportError:
        config_path = Path(__file__).resolve().parents[3] / "olive" / "olive_config.json"

    if config_path and config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def _resolve_quantize_packages(algorithm: str, implementation: str) -> tuple[set[str], list[str]]:
    """Derive quantize dependencies from olive_config.json.

    Returns ``(extras, extra_packages)`` where *extras* are olive-ai extras
    keys (e.g. ``"bnb"``, ``"inc"``) and *extra_packages* are direct pip
    package names (e.g. ``"autoawq"``).
    """
    extras: set[str] = set()
    extra_packages: list[str] = []

    config = _load_olive_config()
    if config is None:
        return extras, extra_packages

    passes_cfg = config.get("passes", {})
    extra_deps_map = config.get("extra_dependencies", {})

    candidate_pass_types = _IMPL_TO_PASS_TYPES.get(implementation, [])
    for pass_type in candidate_pass_types:
        pass_info = passes_cfg.get(pass_type)
        if pass_info is None:
            continue

        # Skip passes that do not support the requested algorithm.
        supported_algos = pass_info.get("supported_algorithms", [])
        if supported_algos and "*" not in supported_algos and algorithm not in supported_algos:
            continue

        # extra_dependencies are olive-ai extras keys (resolved via setup.py)
        for dep_key in pass_info.get("extra_dependencies", []):
            if dep_key in extra_deps_map:
                extras.add(dep_key)
            else:
                extra_packages.append(dep_key)

        # module_dependencies are direct pip packages
        extra_packages.extend(pass_info.get("module_dependencies", []))

        # If the pass may need a dataset, pre-install the datasets package.
        dataset_req = pass_info.get("dataset", "dataset_not_required")
        if dataset_req in ("dataset_required", "dataset_optional", "dataset"):
            extra_packages.append("datasets")

    return extras, extra_packages


def _resolve_packages(command: str, provider: str | None = None, **kwargs) -> list[str]:
    """Resolve all packages needed for a given olive command + options.

    Uses olive-ai[extras] syntax to pull in the right onnxruntime variant,
    plus any additional packages that specific passes/commands require.
    Includes hidden/implicit dependencies that olive_config.json doesn't declare.
    """
    extras = set()
    extra_packages = []

    # 1. Base ORT variant from provider
    if provider:
        ep_extra = PROVIDER_TO_EXTRAS.get(provider)
        if ep_extra:
            extras.add(ep_extra)

    # 2. Command-specific extras
    if command == Command.OPTIMIZE:
        exporter = kwargs.get("exporter") or "model_builder"
        precision = kwargs.get("precision", "fp32")

        if exporter == "model_builder":
            # ModelBuilder pass requires onnxruntime-genai (undeclared in olive_config.json)
            genai_pkg = PROVIDER_TO_GENAI.get(provider or "CPUExecutionProvider", "onnxruntime-genai")
            extra_packages.append(genai_pkg)
        elif exporter == "optimum_exporter":
            extras.add("optimum")

        # GPTQ pass is enabled for int4/uint4 precision → needs calibration data → datasets
        if precision in ("int4", "uint4"):
            extra_packages.append("datasets")

    elif command == Command.QUANTIZE:
        algorithm = kwargs.get("algorithm", "rtn")
        impl = kwargs.get("implementation", "olive")
        q_extras, q_packages = _resolve_quantize_packages(algorithm, impl)
        extras.update(q_extras)
        extra_packages.extend(q_packages)

    elif command == Command.FINETUNE:
        method = kwargs.get("method", "lora")
        if method == "qlora":
            extras.add("finetune")  # includes bnb, peft, accelerate, etc.
        else:
            extras.add("lora")  # peft, accelerate, scipy
        # Fine-tuning always loads datasets
        extra_packages.append("datasets")

    elif command == Command.CAPTURE_ONNX_GRAPH:
        extras.add("capture-onnx-graph")  # optimum only — does NOT include onnxruntime
        # Model builder variant for capture
        if kwargs.get("use_model_builder"):
            genai_pkg = PROVIDER_TO_GENAI.get(provider or "CPUExecutionProvider", "onnxruntime-genai")
            extra_packages.append(genai_pkg)

    elif command == Command.BENCHMARK:
        device = kwargs.get("device", "cpu")
        if device == "gpu":
            extras.add("gpu")
        else:
            extras.add("cpu")
        # lm-eval is the evaluation backend
        extra_packages.extend(["lm_eval", "datasets"])

    elif command == Command.DIFFUSION_LORA:
        extras.add("diffusers")  # accelerate, peft, diffusers
        extra_packages.append("datasets")

    # 3. Ensure a base onnxruntime is always present.
    #    Many olive passes need onnxruntime at runtime even if they don't declare it.
    #    The extras "cpu", "gpu", "directml", etc. each install the right ORT variant.
    #    If none was added yet, default to "cpu" so bare onnxruntime is available.
    ort_extras = {"cpu", "gpu", "directml", "openvino", "qnn"}
    if not extras.intersection(ort_extras):
        extras.add("cpu")

    # 4. Build olive-ai install string with extras
    olive_install = f"olive-ai[{','.join(sorted(extras))}]"

    # 5. Deduplicate extra_packages
    extra_packages = list(set(extra_packages))

    return [olive_install, *extra_packages]
