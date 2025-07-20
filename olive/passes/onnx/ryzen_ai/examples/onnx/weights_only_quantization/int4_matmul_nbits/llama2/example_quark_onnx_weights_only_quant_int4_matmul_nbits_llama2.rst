.. raw:: html

   <!-- omit in toc -->

Quantizating Llama-2-7b model using MatMulNBits quantizer
=========================================================

.. note::

   For information on accessing Quark ONNX examples, refer to `Accessing ONNX Examples <onnx_examples>`_.
   This example and the relevant files are available at ``onnx/weights_only_quantization/int4_matmul_nbits/llama2``

This example describes how to quantize an Llama-2-7b model using the ONNX MatMulNBits quantizer of Quark including an option for HQQ algorithm.

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

   mkdir models && optimum-cli export onnx --model ./Llama-2-7b-hf --task text-generation-with-past ./models/

Convert the float model to float16 due to reduce memory overhead:

::

   mkdir fp16_models
   cp -r models/*.json fp16_models
   cp models/tokenizer.model fp16_models
   python -m quark.onnx.tools.convert_fp32_to_fp16 --input models/model.onnx --output fp16_models/model.onnx --disable_shape_infer --save_as_external_data --all_tensors_to_one_file

Quantization
------------

The quantizer takes the float16 model and produces a matmul_4bits quantized model:

::

   mkdir matmul_4bits_quantized_models
   cp -r fp16_models/*.json matmul_4bits_quantized_models
   cp fp16_models/tokenizer.model matmul_4bits_quantized_models
   python quantize_model.py --input_model_path fp16_models/model.onnx \
                            --output_model_path matmul_4bits_quantized_models/model.onnx \
                            --config MATMUL_NBITS

This command will generate a quantized model under the **matmul_4bits_quantized_models** folder, using the 4bits quantization configuration for MatMul.

Quantization with HQQ
---------------------

The quantizer takes the float16 model and produces a matmul_4bits quantized model with HQQ algorithm:

::

   mkdir matmul_4bits_hqq_quantized_models
   cp -r fp16_models/*.json matmul_4bits_hqq_quantized_models
   cp fp16_models/tokenizer.model matmul_4bits_hqq_quantized_models
   python quantize_model.py --input_model_path fp16_models/model.onnx \
                            --output_model_path matmul_4bits_hqq_quantized_models/model.onnx \
                            --config MATMUL_NBITS \
                            --hqq

This command will generate a quantized model under the **matmul_4bits_hqq_quantized_models** folder, using the 4bits quantization configuration for MatMul with HQQ algorithm.

Evaluation
----------

Test the PPL of the float model on wikitext2.raw:

::

   python onnx_validate.py --model_name_or_path models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda

Test the PPL of the matmul 4bits quantized model:

::

   python onnx_validate.py --model_name_or_path matmul_4bits_quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda

Test the PPL of the matmul 4bits quantized model with hqq algorithm:

::

   python onnx_validate.py --model_name_or_path matmul_4bits_hqq_quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda


+------------+-------------+------------------------------+---------------------------------------+
|            | Float Model | MatMul 4bits Quantized Model | MatMul 4bits Quantized Model with HQQ |
+============+=============+==============================+=======================================+
| Model Size | 26.0 G      | 3.5 G                        | 3.6 G                                 |
+------------+-------------+------------------------------+---------------------------------------+
| PPL        | 5.6267      | 5.9228                       | 5.9210                                |
+------------+-------------+------------------------------+---------------------------------------+

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
