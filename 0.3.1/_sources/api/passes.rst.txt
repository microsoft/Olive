.. _passes:

Passes
=================================
The following passes are available in Olive.

Each pass is followed by a description of the pass and a list of the pass's configuration options.

.. _onnx_conversion:

OnnxConversion
--------------
.. autoconfigclass:: olive.passes.OnnxConversion

.. _device_specific_onnx_conversion:

DeviceSpecificOnnxConversion
----------------------------
.. autoconfigclass:: olive.passes.DeviceSpecificOnnxConversion

.. _onnx_model_optimizer:

OnnxModelOptimizer
------------------
.. autoconfigclass:: olive.passes.OnnxModelOptimizer

.. _ort_transformers_optimization:

OrtTransformersOptimization
----------------------------
.. autoconfigclass:: olive.passes.OrtTransformersOptimization

.. _ort_perf_tuning:

OrtPerfTuning
----------------
.. autoconfigclass:: olive.passes.OrtPerfTuning

.. _onnx_float_to_float16:

OnnxFloatToFloat16
--------------------
.. autoconfigclass:: olive.passes.OnnxFloatToFloat16

.. _ort_mixed_precision:

OrtMixedPrecision
--------------------
.. autoconfigclass:: olive.passes.OrtMixedPrecision

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

.. _inc_dynamic_quantization:

IncDynamicQuantization
-----------------------
.. autoconfigclass:: olive.passes.IncDynamicQuantization

.. _inc_static_quantization:

IncStaticQuantization
----------------------
.. autoconfigclass:: olive.passes.IncStaticQuantization

.. _inc_quantization:

IncQuantization
----------------
.. autoconfigclass:: olive.passes.IncQuantization

.. _append_pre_post_processing:

AppendPrePostProcessingOps
----------------------------
.. autoconfigclass:: olive.passes.AppendPrePostProcessingOps

.. _insert_beam_search:

InsertBeamSearch
--------------------
.. autoconfigclass:: olive.passes.InsertBeamSearch

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

.. _sparsegpt:

SparseGPT
--------------------
.. autoconfigclass:: olive.passes.SparseGPT

.. _torch_trt_conversion:

TorchTRTConversion
--------------------
.. autoconfigclass:: olive.passes.TorchTRTConversion

.. _vitis_ai_quantization:

VitisAIQuantization
--------------------
.. autoconfigclass:: olive.passes.VitisAIQuantization

.. _optimum_conversion:

OptimumConversion
--------------------
.. autoconfigclass:: olive.passes.OptimumConversion

.. _optimum_merging:

OptimumMerging
--------------------
.. autoconfigclass:: olive.passes.OptimumMerging
