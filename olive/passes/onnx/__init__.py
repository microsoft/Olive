# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.passes.onnx.append_pre_post_processing_ops import AppendPrePostProcessingOps
from olive.passes.onnx.bnb_quantization import OnnxBnb4Quantization
from olive.passes.onnx.conversion import OnnxConversion, OnnxOpVersionConversion
from olive.passes.onnx.dynamic_to_fixed_shape import DynamicToFixedShape
from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16
from olive.passes.onnx.genai_model_exporter import GenAIModelExporter
from olive.passes.onnx.inc_quantization import IncDynamicQuantization, IncQuantization, IncStaticQuantization
from olive.passes.onnx.insert_beam_search import InsertBeamSearch
from olive.passes.onnx.mixed_precision import OrtMixedPrecision
from olive.passes.onnx.model_optimizer import OnnxModelOptimizer
from olive.passes.onnx.moe_experts_distributor import MoEExpertsDistributor
from olive.passes.onnx.optimum_conversion import OptimumConversion
from olive.passes.onnx.optimum_merging import OptimumMerging
from olive.passes.onnx.perf_tuning import OrtPerfTuning
from olive.passes.onnx.qnn_preprocess import QNNPreprocess
from olive.passes.onnx.quantization import (
    OnnxDynamicQuantization,
    OnnxMatMul4Quantizer,
    OnnxQuantization,
    OnnxStaticQuantization,
)
from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization
from olive.passes.onnx.vitis_ai_quantization import VitisAIQuantization

__all__ = [
    "AppendPrePostProcessingOps",
    "DynamicToFixedShape",
    "GenAIModelExporter",
    "IncDynamicQuantization",
    "IncQuantization",
    "IncStaticQuantization",
    "InsertBeamSearch",
    "MoEExpertsDistributor",
    "OnnxBnb4Quantization",
    "OnnxConversion",
    "OnnxDynamicQuantization",
    "OnnxFloatToFloat16",
    "OnnxMatMul4Quantizer",
    "OnnxModelOptimizer",
    "OnnxOpVersionConversion",
    "OnnxQuantization",
    "OnnxStaticQuantization",
    "OptimumConversion",
    "OptimumMerging",
    "OrtMixedPrecision",
    "OrtPerfTuning",
    "OrtTransformersOptimization",
    "QNNPreprocess",
    "VitisAIQuantization",
]
