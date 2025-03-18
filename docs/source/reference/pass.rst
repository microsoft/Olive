Passes
=================================

The following passes are available in Olive.

Each pass is followed by a description of the pass and a list of the pass's configuration options.

ONNX
=================================

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

OnnxIODataTypeConverter
------------------------
.. autoconfigclass:: olive.passes.OnnxIODataTypeConverter

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

.. _graph_surgeries:

GraphSurgeries
--------------------
.. autoconfigclass:: olive.passes.GraphSurgeries

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

.. _split_model:

SplitModel
----------
.. autoconfigclass:: olive.passes.SplitModel

.. _static_llm:

StaticLLM
----------
.. autoconfigclass:: olive.passes.StaticLLM

.. _ep_context_binary_generator:

EPContextBinaryGenerator
------------------------
.. autoconfigclass:: olive.passes.EPContextBinaryGenerator

.. _compose_onnx_models:

ComposeOnnxModels
-----------------
.. autoconfigclass:: olive.passes.ComposeOnnxModels

.. _optimum_conversion:

OptimumConversion
-----------------
.. autoconfigclass:: olive.passes.OptimumConversion

.. _optimum_merging:

OptimumMerging
--------------
.. autoconfigclass:: olive.passes.OptimumMerging

.. _model_builder:

ModelBuilder
------------
.. autoconfigclass:: olive.passes.ModelBuilder

Pytorch
=================================

.. _capture_split_info:

CaptureSplitInfo
----------------
.. autoconfigclass:: olive.passes.CaptureSplitInfo

.. _lora:

LoRA
----
.. autoconfigclass:: olive.passes.LoRA

.. _loha:

LoHa
-----
.. autoconfigclass:: olive.passes.LoHa

.. _lokr:

LoKr
-----
.. autoconfigclass:: olive.passes.LoKr

.. _qlora:

QLoRA
-----
.. autoconfigclass:: olive.passes.QLoRA

.. _dora:

DoRA
-----
.. autoconfigclass:: olive.passes.DoRA

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

.. _spinquant:

SpinQuant
---------
.. autoconfigclass:: olive.passes.SpinQuant

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

OpenVINO
=================================

.. _openvino_conversion:

OpenVINOConversion
------------------
.. autoconfigclass:: olive.passes.OpenVINOConversion

.. _openvino_quantization:

OpenVINOQuantization
--------------------
.. autoconfigclass:: olive.passes.OpenVINOQuantization

SNPE
=================================

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

QNN
=================================

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
