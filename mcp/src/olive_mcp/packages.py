# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive_mcp.constants import (
    PROVIDER_TO_EXTRAS,
    PROVIDER_TO_GENAI,
    Command,
)


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
        if impl == "bnb":
            extras.add("bnb")
        elif impl == "inc":
            extras.add("inc")
        elif impl == "autogptq" or algorithm == "gptq":
            extra_packages.extend(["auto-gptq", "optimum", "datasets"])
        elif impl == "awq" or algorithm == "awq":
            extra_packages.append("autoawq")
        # Static quantization needs calibration data
        if algorithm != "rtn":
            extra_packages.append("datasets")

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
