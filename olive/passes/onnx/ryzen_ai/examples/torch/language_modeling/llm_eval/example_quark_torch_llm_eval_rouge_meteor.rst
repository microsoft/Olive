Rouge & Meteor Evaluations
==========================

Below details how to run ROUGE and METEOR evaluations. ROUGE and METEOR
scores are currently available for the following datasets
``[samsum, xsum, cnn_dm]``, where ``cnn_dm`` is an abbreviation for
``cnn_dailymail``.

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

-  The ``--rouge`` and ``--meteor`` specify the rouge and meteor task,
   respectively. You can run either or both.
-  The ``--num_eval_data`` arg is used to specify the number of samples
   used from an eval dataset.
-  The ``--dataset`` arg specifies the dataset. Select from
   ``[xsum, cnn_dm, samsum]``. Can specify multiple as comma-seperated:
   ``--dataset samsum,xsum``.

Rouge/Meteor on Torch Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Rouge and Meteor on 20 samples of XSUM, using a pretrained LLM.
   Example with ``Llama2-7b-hf``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --rouge \
         --meteor \
         --dataset xsum \
         --trust_remote_code \
         --batch_size 1 \
         --num_eval_data 20 \
         --device cuda

Alternatively, to load a local checkpoint:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=[local checkpoint path] \
         --rouge \
         --meteor \
         --dataset xsum \
         --trust_remote_code \
         --batch_size 1 \
         --num_eval_data 20 \
         --device cuda

2. Rouge and Meteor on a Quark Quantized model. Example with
   ``Llama-2-7b-chat-hf-awq-uint4-asym-g128-bf16-lmhead``

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --model_reload \
         --import_file_format hf_format \
         --import_model_dir [path to Llama-2-7b-chat-hf-awq-uint4-asym-g128-bf16-lmhead model] \
         --rouge \
         --meteor \
         --dataset xsum \
         --trust_remote_code \
         --batch_size 1 \
         --num_eval_data 20 \
         --device cuda

Rouge/Meteor on ONNX Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Rouge and Meteor on pretrained, ONNX Exported LLM: Example with
   ``Llama2-7b-hf``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --import_file_format onnx_format \
         --import_model_dir [path to Llama-2-7b-hf ONNX model file] \
         --rouge \
         --meteor \
         --dataset xsum \
         --trust_remote_code \
         --batch_size 1 \
         --num_eval_data 20 \
         --device cpu

2. Rouge and Meteor on Quark Quantized, ONNX Exported LLM: Example with
   ``Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --import_file_format onnx_format \
         - import_model_dir [path to Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx model file] \
         --rouge \
         --meteor \
         --dataset xsum \
         --trust_remote_code \
         --batch_size 1 \
         --num_eval_data 20 \
         --device cpu

Other Arguments
---------------

1. Set ``--multi_gpu`` for multi-gpu support.
2. Set ``--save_metrics_to_csv`` and ``metrics_output_dir`` to save
   scores to CSV.
3. Set ``dtype`` by ``model_args dtype=float32`` to change model dtype.
4. Set ``--seq_len`` for max sequence length on inputs.
5. Set ``--max_new_toks`` for max number of new tokens generated
   (excluding length of input tokens).

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
