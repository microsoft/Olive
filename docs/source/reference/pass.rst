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

OnnxScriptFusion
----------------
.. autoconfigclass:: olive.passes.OnnxScriptFusion

.. _onnxscript_fusion:

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

.. _onnx_quantization_preprocess:

OnnxQuantizationPreprocess
--------------------------
.. autoconfigclass:: olive.passes.OnnxQuantizationPreprocess

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

.. _onnx_rtn_quantization:

OnnxBlockWiseRtnQuantization
----------------------------
.. autoconfigclass:: olive.passes.OnnxBlockWiseRtnQuantization

.. _onnx_kquant_quantization:

OnnxKQuantQuantization
-----------------------
.. autoconfigclass:: olive.passes.OnnxKQuantQuantization

.. _onnx_hqq_quantization:

OnnxHqqQuantization
--------------------
.. autoconfigclass:: olive.passes.OnnxHqqQuantization

.. _graph_surgeries:

GraphSurgeries
--------------------
.. autoconfigclass:: olive.passes.GraphSurgeries

.. _nvmodelopt_graph_surgery:

NVModelOptGraphSurgery
----------------------
.. autoconfigclass:: olive.passes.NVModelOptGraphSurgery

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

Generates an EPContext ONNX model and its compiled context data. QNN uses session-based
compilation and retains support for composite models and weight sharing. OpenVINO uses
the ``onnxruntime_ep_openvino`` plugin and accepts a single ONNX model; no intermediate
OpenVINO IR (``.xml`` and weights ``.bin``) is required.

Install compatible ONNX Runtime and OpenVINO EP plugin builds before using the OpenVINO
path. The default ``compile_flow: aot`` requires ORT's ``ModelCompiler`` API; ``jit``
creates an ``InferenceSession`` to trigger compilation instead.

.. code-block:: bash

   pip install onnxruntime onnxruntime-ep-openvino

Example OpenVINO pass configuration (with ``OpenVINOExecutionProvider`` on an NPU accelerator):

.. code-block:: json

   {
     "type": "EPContextBinaryGenerator",
     "compile_flow": "aot",
     "embed_context": false,
     "disable_cpu_fallback": true,
     "ep_registration_name": "ovep_registration.virtual",
     "ep_device_filters": {"npu_platform": "5010"},
     "provider_options": {},
     "session_options": {"log_severity_level": 1},
     "model_compiler_options": {
       "graph_optimization_level": "ORT_DISABLE_ALL",
       "flags": "ERROR_IF_NO_NODES_COMPILED"
     }
   }

The plugin registration name defaults to its EP name for local devices. A name ending
in ``.virtual`` exposes virtual cross-compilation devices when supported by the plugin.
Selection is restricted to the configured accelerator type and must identify exactly one
device; use ``ep_device_filters`` to match device metadata or a hexadecimal ``device_id``.

``embed_context`` controls both the EPContext session setting and the AOT compiler's
embedding option. An explicit ``model_compiler_options.embed_compiled_data_into_model``
overrides it for AOT. Compiler options are not accepted for JIT.
The pass controls EPContext export, embedding, and its output path through the compile API
for AOT or session entries for JIT. Other session entries are forwarded as strings, except
the typed logging attributes. With external context data,
the returned model resource is a directory containing the EPContext ONNX and its binaries.

This pass only compiles models: it does not assemble a model package or enable weightless
compilation by default. If ``ep.enable_weightless`` is enabled explicitly, retain the original
weights and configure their location when loading the compiled model.

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

.. _aimet_quantization:

AimetQuantization
-----------------

.. autoconfigclass:: olive.passes.AimetQuantization

.. _quark_quantization:

QuarkQuantization
-----------------

.. autoconfigclass:: olive.passes.QuarkQuantization

Pytorch
=================================

.. _capture_split_info:

CaptureSplitInfo
----------------
.. autoconfigclass:: olive.passes.CaptureSplitInfo

.. _selective_mixed_precision:

SelectiveMixedPrecision
-----------------------
.. autoconfigclass:: olive.passes.SelectiveMixedPrecision

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

.. _gptq:

Gptq
-----
.. autoconfigclass:: olive.passes.Gptq

.. _gptq_quantizer:

GptqQuantizer
-------------
.. autoconfigclass:: olive.passes.GptqQuantizer

.. _awq_quantizer:

AutoAWQQuantizer
----------------
.. autoconfigclass:: olive.passes.AutoAWQQuantizer

.. _rtn:

Rtn
----------------
.. autoconfigclass:: olive.passes.Rtn

OpenVINO
=================================

.. _openvino_conversion:

OpenVINOConversion
------------------
.. autoconfigclass:: olive.passes.OpenVINOConversion

.. _openvino_ioupdate:

OpenVINOIoUpdate
------------------
.. autoconfigclass:: olive.passes.OpenVINOIoUpdate

.. _openvino_quantization:

OpenVINOQuantization
--------------------
.. autoconfigclass:: olive.passes.OpenVINOQuantization

.. _openvino_quantization_with_accuracy:

OpenVINOQuantizationWithAccuracy
--------------------------------
.. autoconfigclass:: olive.passes.OpenVINOQuantizationWithAccuracy

.. _openvino_weight_compression:

OpenVINOWeightCompression
--------------------------
.. autoconfigclass:: olive.passes.OpenVINOWeightCompression

.. _openvino_encapsulation:

OpenVINOEncapsulation
----------------------
.. autoconfigclass:: olive.passes.OpenVINOEncapsulation

.. _openvino_optimum_conversion:

OpenVINOOptimumConversion
-------------------------
.. autoconfigclass:: olive.passes.OpenVINOOptimumConversion

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
