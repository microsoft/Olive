.. raw:: html

   <!-- omit in toc -->

Dynamic Quantization for OPT-125M
=================================

This folder contains an example of quantizing an opt-125m model using the ONNX quantizer of Quark.
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

Get opt-125m torch model:

::

   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/tokenizer_config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/vocab.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/merges.txt
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/generation_config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/special_tokens_map.json

Export onnx model from opt-125m torch model:

::

   mkdir models && optimum-cli export onnx --model ./opt-125m --task text-generation ./models/

Quantization
------------

The quantizer takes the float model and produces a dynamically quantized model:

::

   cp -r models dynamic_quantized_models && rm dynamic_quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path dynamic_quantized_models/dynamic_quantized_model.onnx \
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
| Model Size | 480 MB      | 120 MB                  |
+------------+-------------+-------------------------+
| PPL        | 27.0317     | 28.6006                 |
+------------+-------------+-------------------------+

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
