#!/usr/bin/env python3
"""Verify OnnxKQuantQuantization results in an Olive ORT-GenAI package.

For each *.onnx file in a package directory, counts:
  - MatMul        : float (un-quantized) matmuls
  - MatMulNBits   : int4/int-N quantized matmuls (com.microsoft) produced by
                    OnnxKQuantQuantization

Usage:
    python verify_quantization.py /path/to/package_dir [more_dirs...]

Interpretation:
  - "Quantized correctly" for a component  -> MatMulNBits > 0 and MatMul (with
    initializer weight) ~ 0.
  - "Excluded / not quantized"             -> MatMulNBits == 0, MatMul > 0.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import onnx


def _walk(graph, counts: Counter) -> None:
    for node in graph.node:
        counts[node.op_type] += 1
        for attr in node.attribute:
            if attr.HasField("g"):
                _walk(attr.g, counts)
            for g in attr.graphs:
                _walk(g, counts)


def count_ops(model_path: Path) -> Counter:
    # load_external_data=False: we only need the graph structure (op types),
    # not the (large) external weight tensors.
    model = onnx.load(str(model_path), load_external_data=False)
    counts: Counter = Counter()
    _walk(model.graph, counts)
    return counts


def report(pkg: Path) -> None:
    print(f"\n=== Package: {pkg} ===")
    onnx_files = sorted(pkg.glob("*.onnx"))
    if not onnx_files:
        print("  (no .onnx files found)")
        return
    for f in onnx_files:
        counts = count_ops(f)
        mm = counts.get("MatMul", 0)
        mmnbits = counts.get("MatMulNBits", 0)
        gemm = counts.get("Gemm", 0)
        status = "QUANTIZED" if mmnbits > 0 else "not quantized"
        print(f"\n  {f.name}  [{status}]")
        print(f"    MatMulNBits (int4 quantized) : {mmnbits}")
        print(f"    MatMul      (float)          : {mm}")
        if gemm:
            print(f"    Gemm        (float)          : {gemm}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        report(Path(d))


if __name__ == "__main__":
    main()
