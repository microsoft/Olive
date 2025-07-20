.. raw:: html

   <!-- omit in toc -->

Quark ONNX Quantization Example
===============================

This folder contains an example of quantizing an Llama-2-7b model using the ONNX  quantizer of Quark.
The example has the following parts:

- `Pip Requirements <#pip-requirements>`__
- `Prepare Model <#prepare-model>`__
- `Quantization <#quantization>`__
- `Evaluation <#evaluation>`__


Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r requirements.txt

Prepare model
-------------
Download the HF Llama-2-7b-hf checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta.
For additional details, see the `Llama2 page on Huggingface <https://huggingface.co/docs/transformers/main/en/model_doc/llama2>`__. Upon obtaining permission, download the checkpoint to the **Llama-2-7b-hf** folder.


Export onnx model from Llama-2-7b-hf torch model:

::

   mkdir models && optimum-cli export onnx --model ./Llama-2-7b-hf --task text-generation ./models/

Convert the float model to float16 due to reduce memory overhead:

::

   mkdir fp16_models
   cp -r models/*.json fp16_models
   cp models/tokenizer.model fp16_models
   python -m quark.onnx.tools.convert_fp32_to_fp16 --input models/model.onnx --output fp16_models/model.onnx --disable_shape_infer --save_as_external_data --all_tensors_to_one_file

Quantization
------------

The quantizer takes the float16 model and produces a matmul int8 weights only quantized model:

::

   mkdir quantized_models
   cp -r fp16_models/*.json quantized_models
   cp fp16_models/tokenizer.model quantized_models
   python quantize_model.py --input_model_path fp16_models/model.onnx \
                            --output_model_path quantized_models/model.onnx \
                            --config INT8_TRANSFORMER_DEFAULT

This command will generate a quantized model under the **quantized_models** folder, using the 8bits weights only quantization configuration for MatMul.

Evaluation
----------

Test the PPL of the float model on wikitext2.raw:

::

   python onnx_validate.py --model_name_or_path models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda

Test the PPL of the quantized model:

::

   python onnx_validate.py --model_name_or_path quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda


+------------+-------------+-----------------+
|            | Float Model | Quantized Model |
+============+=============+=================+
| Model Size | 26.0 G      | 6.7 G           |
+------------+-------------+-----------------+
| PPL        | 5.6267      | 5.8255          |
+------------+-------------+-----------------+

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
