.. raw:: html

   <!-- omit in toc -->

Quantizing an OPT-125M Model
============================

.. note::

   For information on accessing Quark ONNX examples, refer to `Accessing ONNX Examples <onnx_examples>`_.
   This example and the relevant files are available at ``onnx/weights_only_quantization/int8_qdq/llama2``

This example describes how to quantize an opt-125m model using the ONNX quantizer of Quark.


Pip requirements
----------------

Install the necessary Python packages:

::

   python -m pip install -r requirements.txt

Prepare model
-------------

Get opt-125m torch model:

::

   mkdir opt-125m
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/tokenizer_config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/vocab.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/merges.txt
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/generation_config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/special_tokens_map.json

Export ONNX model from opt-125m torch model:

::

   mkdir models && optimum-cli export onnx --model ./opt-125m --task text-generation ./models/

Quantization
------------

The quantizer takes the float model and produces a quantized model.

::

   cp -r models quantized_models && rm quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path quantized_models/quantized_model.onnx \
                            --config INT8_TRANSFORMER_DEFAULT

This command will generate a quantized model under the **quantized_models** folder, which was quantized by Int8 configuration for transformer-based models.

Evaluation
----------

Test the PPL of the float model on wikitext2.raw:

::

   python onnx_validate.py --model_name_or_path models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model models/ --do_onnx_eval --no_cuda

Test the PPL of the quantized model:

::

   python onnx_validate.py --model_name_or_path quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model quantized_models/ --do_onnx_eval --no_cuda

+-------+--------------------+---------------------+
|       | Float Model        | Quantized Model     |
+=======+====================+=====================+
| Model | 480 MB             | 384 MB              |
| Size  |                    |                     |
+-------+--------------------+---------------------+
| PPL   | 27.0317            | 28.6846             |
+-------+--------------------+---------------------+

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
