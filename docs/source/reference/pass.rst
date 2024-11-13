Passes
=================================

The following passes are available in Olive.

Each pass is followed by a description of the pass and a list of the pass's configuration options.

.. _onnx_conversion:

OnnxConversion
--------------
.. autoconfigclass:: olive.passes.OnnxConversion

.. _onnx_op_version_conversion:

OnnxOpVersionConversion
-----------------------
.. autoconfigclass:: olive.passes.OnnxOpVersionConversion

.. _onnx_peephole_optimizer:

OnnxPeepholeOptimizer
---------------------
.. autoconfigclass:: olive.passes.OnnxPeepholeOptimizer

.. _ort_transformers_optimization:

OrtTransformersOptimization
---------------------------
.. autoconfigclass:: olive.passes.OrtTransformersOptimization

.. _ort_session_params_tuning:

OrtSessionParamsTuning
----------------------
.. autoconfigclass:: olive.passes.OrtSessionParamsTuning

.. _onnx_float_to_float16:

OnnxFloatToFloat16
------------------
.. autoconfigclass:: olive.passes.OnnxFloatToFloat16

.. _onnx_io_float16_to_float32:

OnnxIOFloat16ToFloat32
----------------------
.. autoconfigclass:: olive.passes.OnnxIOFloat16ToFloat32

.. _ort_mixed_precision:

OrtMixedPrecision
-----------------
.. autoconfigclass:: olive.passes.OrtMixedPrecision

.. _qnn_preprocess:

QNNPreprocess
-------------
.. autoconfigclass:: olive.passes.QNNPreprocess

.. _mixed_precision_overrides:

MixedPrecisionOverrides
-----------------------
.. autoconfigclass:: olive.passes.MixedPrecisionOverrides

.. _onnx_dynamic_quantization:

OnnxDynamicQuantization
-----------------------
.. autoconfigclass:: olive.passes.OnnxDynamicQuantization

.. _onnx_static_quantization:

OnnxStaticQuantization
----------------------
.. autoconfigclass:: olive.passes.OnnxStaticQuantization

.. _onnx_quantization:

OnnxQuantization
----------------
.. autoconfigclass:: olive.passes.OnnxQuantization

.. _onnx_matmul4_quantizer:

OnnxMatMul4Quantizer
--------------------
.. autoconfigclass:: olive.passes.OnnxMatMul4Quantizer

.. _matmulnbits_to_qdq:

MatMulNBitsToQDQ
----------------
.. autoconfigclass:: olive.passes.MatMulNBitsToQDQ

.. _dynamic_to_fixed_shape:

DynamicToFixedShape
-------------------
.. autoconfigclass:: olive.passes.DynamicToFixedShape

.. _inc_dynamic_quantization:

IncDynamicQuantization
----------------------
.. autoconfigclass:: olive.passes.IncDynamicQuantization

.. _inc_static_quantization:

IncStaticQuantization
---------------------
.. autoconfigclass:: olive.passes.IncStaticQuantization

.. _inc_quantization:

IncQuantization
---------------
.. autoconfigclass:: olive.passes.IncQuantization

.. _vitis_ai_quantization:

VitisAIQuantization
-------------------
.. autoconfigclass:: olive.passes.VitisAIQuantization

.. _append_pre_post_processing:

AppendPrePostProcessingOps
--------------------------
.. autoconfigclass:: olive.passes.AppendPrePostProcessingOps

.. _insert_beam_search:

InsertBeamSearch
----------------
.. autoconfigclass:: olive.passes.InsertBeamSearch

.. _extract_adapters:

ExtractAdapters
---------------
.. autoconfigclass:: olive.passes.ExtractAdapters

.. _capture_split_info:

CaptureSplitInfo
----------------
.. autoconfigclass:: olive.passes.CaptureSplitInfo

.. _split_model:

SplitModel
----------
.. autoconfigclass:: olive.passes.SplitModel

.. _lora:

LoRA
----
.. autoconfigclass:: olive.passes.LoRA

.. _qlora:

QLoRA
-----
.. autoconfigclass:: olive.passes.QLoRA

.. _loftq:

LoftQ
-----
.. autoconfigclass:: olive.passes.LoftQ

.. _lora_hf_training_arguments:

LoRA/QLoRA/LoftQ HFTrainingArguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autopydantic_settings:: olive.passes.pytorch.lora.HFTrainingArguments

.. _quantization_aware_training:

QuantizationAwareTraining
-------------------------
.. autoconfigclass:: olive.passes.QuantizationAwareTraining

.. _openvino_conversion:

OpenVINOConversion
------------------
.. autoconfigclass:: olive.passes.OpenVINOConversion

.. _openvino_quantization:

OpenVINOQuantization
--------------------
.. autoconfigclass:: olive.passes.OpenVINOQuantization

.. _snpe_conversion:

SNPEConversion
--------------
.. autoconfigclass:: olive.passes.SNPEConversion

.. _snpe_quantization:

SNPEQuantization
----------------
.. autoconfigclass:: olive.passes.SNPEQuantization

.. _snpe_to_onnx_conversion:

SNPEtoONNXConversion
--------------------
.. autoconfigclass:: olive.passes.SNPEtoONNXConversion

.. _qnn_conversion:

QNNConversion
-------------
.. autoconfigclass:: olive.passes.QNNConversion

.. _qnn_model_lib_generator:

QNNModelLibGenerator
--------------------
.. autoconfigclass:: olive.passes.QNNModelLibGenerator

.. _qnn_context_binary_generator:

QNNContextBinaryGenerator
-------------------------
.. autoconfigclass:: olive.passes.QNNContextBinaryGenerator

.. _merge_adapter_weights:

MergeAdapterWeights
-------------------
.. autoconfigclass:: olive.passes.MergeAdapterWeights

.. _sparsegpt:

SparseGPT
---------
.. autoconfigclass:: olive.passes.SparseGPT

.. _slicegpt:

SliceGPT
--------
.. autoconfigclass:: olive.passes.SliceGPT

.. _quarot:

QuaRot
------
.. autoconfigclass:: olive.passes.QuaRot

.. _gptq_quantizer:

GptqQuantizer
-------------
.. autoconfigclass:: olive.passes.GptqQuantizer

.. _awq_quantizer:

AutoAWQQuantizer
----------------
.. autoconfigclass:: olive.passes.AutoAWQQuantizer

.. _torch_trt_conversion:

TorchTRTConversion
------------------
.. autoconfigclass:: olive.passes.TorchTRTConversion

.. _optimum_conversion:

OptimumConversion
-----------------
.. autoconfigclass:: olive.passes.OptimumConversion

.. _optimum_merging:

OptimumMerging
--------------
.. autoconfigclass:: olive.passes.OptimumMerging

.. model_builder:

ModelBuilder
------------
.. autoconfigclass:: olive.passes.ModelBuilder
