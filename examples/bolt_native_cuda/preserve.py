# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Reference a native Bolt package through Olive without transforming its graphs or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import onnx_ir as ir

from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.workflows.run.config import RunConfig

COMPONENTS = {"decoder": "decoder", "vision_encoder": "vision", "embedding": "embedding"}
PACKED_OPS = {"MatMulGGMLIQ2XXS", "MatMulGGMLQ8_0", "MatMulNVFP4A16"}
TEMPLATE = Path(__file__).with_name("olive_config.json")


def sha256(path: Path) -> str:
    """Hash without loading a weight shard into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_file(root: Path, relative: str) -> Path:
    """Resolve package-local paths, rejecting escapes and missing files."""
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError(f"Path must stay inside the native package: {relative}")
    if not resolved.is_file():
        raise ValueError(f"Missing package file: {relative}")
    return resolved


def inspect_package(package_dir: Path, config_path: Path = TEMPLATE, route: str = "prepared-features"):
    """Create existing Olive handlers and inspect graph/external-data references lazily."""
    if route != "prepared-features":
        raise ValueError(
            "Only prepared-features is supported; stock ORT GenAI/ort-extensions routing is not supported."
        )
    root = package_dir.resolve(strict=True)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("passes") != {} or raw.get("builds"):
        raise ValueError("Native preservation forbids passes/builds, including post-export requantization.")
    config = raw["input_model"]["config"]
    if raw["input_model"]["type"].lower() != "compositemodel":
        raise ValueError("Native preservation requires CompositeModel.")
    if config["model_component_names"] != list(COMPONENTS) or len(config["model_components"]) != len(COMPONENTS):
        raise ValueError("Expected decoder, vision_encoder, embedding components in that order.")
    config["model_path"] = str(root)
    genai_path = package_file(root, "genai_config.json")
    genai = json.loads(genai_path.read_text(encoding="utf-8"))
    files = {genai_path}
    inventory = {}
    graph_metadata = {}
    for component_config, (name, genai_name) in zip(config["model_components"], COMPONENTS.items()):
        if component_config["type"].lower() != "onnxmodel":
            raise ValueError("Native components must use ONNXModel.")
        child = component_config["config"]
        expected = f"{name}/model.onnx"
        if child.get("onnx_file_name") != expected or genai["model"][genai_name]["filename"] != expected:
            raise ValueError(f"Component/GenAI filename mismatch for {name}.")
        child["model_path"] = str(root)
        child["inference_settings"] = {"execution_provider": ["CUDAExecutionProvider"]}
        providers = genai["model"][genai_name].get("session_options", {}).get("provider_options", [])
        if providers:
            if len(providers) != 1 or set(providers[0]) != {"cuda"}:
                raise ValueError(f"Only CUDA provider options are supported for {name}.")
            child["inference_settings"]["provider_options"] = [providers[0]["cuda"]]
        if child.get("use_ort_extensions"):
            raise ValueError("No ort-extensions processor is supported by this recipe.")
        graph_path = package_file(root, expected)
        files.add(graph_path)
        # IR loading keeps external tensors as references; never call numpy()/tobytes() or save().
        model = ir.load(graph_path)
        graph_metadata[name] = dict(model.metadata_props)
        loading = json.loads(model.metadata_props.get("mobius.weight_loading", "{}"))
        if (
            loading.get("revision") != config["model_attributes"]["source_revision"]
            or loading.get("full_checkpoint_dequantization") is not False
            or loading.get("native_iq2_storage") is not True
        ):
            raise ValueError(f"Missing or inconsistent native weight-loading provenance in {name}.")
        operators = Counter(f"{node.domain}::{node.op_type}" for node in model.graph.all_nodes())
        if name == "decoder" and not all(f"com.microsoft::{op}" in operators for op in PACKED_OPS):
            raise ValueError("Decoder must retain IQ2_XXS, Q8_0 and NVFP4A16 packed operators.")
        external_count = 0
        for graph in [model.graph, *model.graph.subgraphs()]:
            for value in graph.initializers.values():
                tensor = value.const_value
                if isinstance(tensor, ir.ExternalTensor):
                    relative = (Path(expected).parent / tensor.location).as_posix()
                    data = package_file(root, relative)
                    offset = tensor.offset or 0
                    length = tensor.length if tensor.length is not None else tensor.nbytes
                    if offset < 0 or length < 0 or offset + length > data.stat().st_size:
                        raise ValueError(f"External tensor outside payload bounds: {name}/{value.name}")
                    files.add(data)
                    external_count += 1
        inventory[name] = {"operators": dict(sorted(operators.items())), "external_tensors": external_count}
    # Preserve tokenizer, processing configs, chat template and other root assets as references.
    assets = sorted(path for path in root.iterdir() if path.is_file())
    for path in assets:
        files.add(package_file(root, path.name))
    attributes = config["model_attributes"]
    attributes.update(
        additional_files=[str(path) for path in assets],
        genai_config=genai,
        graph_metadata=graph_metadata,
        model_readiness="NOT_ESTABLISHED",
    )
    validated = RunConfig.model_validate(raw)
    handler = validated.input_model.create_model()
    # Exercise serialization/reloading so all propagated attributes are retained in the reference manifest.
    model_config = ModelConfig.model_validate(handler.to_json())
    model_config.create_model()
    return model_config, inventory, sorted(files)


def check_runtime(model_config: ModelConfig):
    """Load each original component with CUDA; this is not inference or accuracy validation."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable; CPU fallback is not a supported native Bolt route.")
    for name, component in model_config.create_model().get_model_components():
        try:
            session = component.prepare_session(device=Device.GPU, execution_providers=["CUDAExecutionProvider"])
        except Exception as exc:
            raise RuntimeError(
                f"Native CUDA session creation failed for {name}. Install a runtime with the packed operators, "
                "LinearAttention, compatible CUDA libraries and BF16 support; requantization is not a fallback."
            ) from exc
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(f"Native CUDA session creation failed for {name}: CUDA provider failed to initialize.")
        del session
    return {
        "status": "SESSION_CREATION_ONLY",
        "version": ort.__version__,
        "import": str(Path(ort.__file__).resolve()),
        "providers": ort.get_available_providers(),
    }


def preserve(package_dir: Path, output_dir: Path, config_path: Path = TEMPLATE, inspect_only: bool = False, **kwargs):
    """Write only reference metadata and hashes, never a model or external-data copy."""
    root = package_dir.resolve(strict=True)
    output = output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the read-only native package.")
    model_config, inventory, files = inspect_package(root, config_path, **kwargs)
    runtime = {"status": "NOT_CHECKED"} if inspect_only else check_runtime(model_config)
    receipt = {
        "operation": "native-preservation-reference-only",
        "model_readiness": "NOT_ESTABLISHED",
        "accuracy": "NOT_EVALUATED",
        "optimization": "NONE",
        "runtime": runtime,
        "components": inventory,
        "files": {str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256(path)} for path in files},
    }
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive writes also prevent following pre-existing output symlinks into the source package.
    with (output / "model_config.json").open("x", encoding="utf-8") as stream:
        json.dump(model_config.to_json(), stream, indent=2)
    with (output / "preservation_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def main():
    """Run the bounded native-preservation example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=TEMPLATE)
    parser.add_argument("--route", default="prepared-features")
    parser.add_argument("--inspect-only", action="store_true", help="Explicitly skip runtime checks; no runtime claim.")
    args = parser.parse_args()
    receipt = preserve(args.package_dir, args.output_dir, args.config, args.inspect_only, route=args.route)
    sys.stdout.write(
        json.dumps(
            {"operation": receipt["operation"], "runtime": receipt["runtime"], "model_readiness": "NOT_ESTABLISHED"}
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
