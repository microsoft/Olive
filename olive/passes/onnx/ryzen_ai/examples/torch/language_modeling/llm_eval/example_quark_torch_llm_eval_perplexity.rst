Perplexity Evaluations
======================

Below details how to run perplexity evaluations. Perplexity evaluations
utilize the wikitext2 dataset. Supported devices currently are CPU and
GPU.

Summary of support:

+---------+-----------+------------+------------+---------+---------+---------+
| Model   | Quark     | Pretrained | Perplexity | ROUGE   | METEOR  | `LM     |
| Types   | Quantized |            |            |         |         | Eval    |
|         |           |            |            |         |         | Harness |
|         |           |            |            |         |         | Tasks   |
|         |           |            |            |         |         | <ht     |
|         |           |            |            |         |         | tps://g |
|         |           |            |            |         |         | ithub.c |
|         |           |            |            |         |         | om/Eleu |
|         |           |            |            |         |         | therAI/ |
|         |           |            |            |         |         | lm-eval |
|         |           |            |            |         |         | uation- |
|         |           |            |            |         |         | harness |
|         |           |            |            |         |         | /tree/m |
|         |           |            |            |         |         | ain>`__ |
+=========+===========+============+============+=========+=========+=========+
| *LLMs*  |           |            |            |         |         |         |
+---------+-----------+------------+------------+---------+---------+---------+
| - Torch | ✓         | ✓          | ✓          | ✓       | ✓       | ✓       |
+---------+-----------+------------+------------+---------+---------+---------+
| - ONNX  | ✓         | ✓          | ✓          | ✓       | ✓       | ✓       |
+---------+-----------+------------+------------+---------+---------+---------+
| *VLMs*  |           |            |            |         |         |         |
+---------+-----------+------------+------------+---------+---------+---------+
| - Torch | ✓         | ✓          | ✓          | ✓       | ✓       | ✓       |
+---------+-----------+------------+------------+---------+---------+---------+
| - ONNX  | X         | X          | X          | X       | X       | X       |
+---------+-----------+------------+------------+---------+---------+---------+

Recipes
-------

-  The ``--ppl`` argument specifies the perplexity task.

PPL on Torch Models
~~~~~~~~~~~~~~~~~~~

1. PPL on a pretrained LLM. Example with ``Llama2-7b-hf``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --ppl \
         --trust_remote_code \
         --batch_size 1 \
         --device cuda

Alternatively, to load a local checkpoint:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=[local checkpoint path] \
         --ppl \
         --trust_remote_code \
         --batch_size 1 \
         --device cuda

2. PPL on a Quark Quantized model. Example with
   ``Llama-2-7b-chat-hf-awq-uint4-asym-g128-bf16-lmhead``

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --model_reload \
         --import_file_format hf_format \
         --import_model_dir [path to Llama-2-7b-chat-hf-awq-uint4-asym-g128-bf16-lmhead model] \
         --ppl \
         --trust_remote_code \
         --batch_size 1 \
         --device cuda

Perplexity on ONNX Models
~~~~~~~~~~~~~~~~~~~~~~~~~

1. PPL on a pretrained LLM. Example with ``Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --import_file_format onnx_format \
         --import_model_dir [path to Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx model] \
         --ppl \
         --trust_remote_code \
         --batch_size 1 \
         --device cuda

Other Arguments
---------------

1. Set ``--multi_gpu`` for multi-gpu support.
2. Set ``--save_metrics_to_csv`` and ``metrics_output_dir`` to save PPL
   score to CSV.
3. Set ``dtype`` by ``model_args dtype=float32`` to change model dtype.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
