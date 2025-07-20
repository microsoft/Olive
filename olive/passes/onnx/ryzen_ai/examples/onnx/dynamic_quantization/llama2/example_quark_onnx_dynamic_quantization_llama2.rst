.. raw:: html

   <!-- omit in toc -->

Dynamic Quantization for Llama-2-7b
===================================

This folder contains an example of quantizing an Llama-2-7b model using the ONNX quantizer of Quark.
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

Quantization
------------

The quantizer takes the float model and produces a dynamically quantized model:

::

   cp -r models dynamic_quantized_models
   find dynamic_quantized_models -type f -name '*onnx*' -exec rm -f {} \;
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path dynamic_quantized_models/model.onnx \
                            --config UINT8_DYNAMIC_QUANT

This command will generate a quantized model under the **dynamic_quantized_models** folder, using the UInt8 dynamic quantization configuration.

Evaluation
----------

Test the PPL of the float model on wikitext2.raw:

::

   python onnx_validate.py --model_name_or_path models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda

Test the PPL of the dynamic quantized model:

::

   python onnx_validate.py --model_name_or_path dynamic_quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --do_onnx_eval --no_cuda


+------------+-------------+-------------------------+
|            | Float Model | Dynamic Quantized Model |
+============+=============+=========================+
| Model Size | 26.0 G      | 6.4 G                   |
+------------+-------------+-------------------------+
| PPL        | 5.6267      | 8.8859                  |
+------------+-------------+-------------------------+

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
