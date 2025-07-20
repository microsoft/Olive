LM-Evaluation-Harness Evaluations
=================================

Below details how to run evaluations on
`LM-Evaluation-Harness <https://github.com/EleutherAI/lm-evaluation-harness/tree/main>`__
tasks.

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

-  The ``--model hf`` arg is used to run lm-evaluation-harness on all huggingface
   LLMs.
-  The ``--model hf multimodal`` arg is used to run lm-evaluation-harness on
   supported VLMs. We currently support
   ``["Llama-3.2-11B-Vision", "Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct"]``.
-  The ``--tasks`` arg is used to specify dataset of choice. See
   `here <https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks>`__
   for supported tasks from lm-eval-harness.

LM-Evaluation-Harness on Torch Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. LM-Evaluation-Harness, using a pretrained LLM. Example with
   ``Llama2-7b-hf``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --model hf \
         --tasks mmlu_management \
         --batch_size 1 \
         --device cuda

Alternatively, to load a local checkpoint:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=[local checkpoint path] \
         --model hf \
         --tasks mmlu_management \
         --batch_size 1 \
         --device cuda

2. LM-Evaluation-Harness on a Quark Quantized model. Example with
   ``Llama-2-7b-chat-hf-awq-uint4-asym-g128-bf16-lmhead``

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --model_reload \
         --import_file_format hf_format \
         --import_model_dir [path to Llama-2-7b-chat-hf-awq-uint4-asym-g128-bf16-lmhead model] \
         --model hf \
         --tasks mmlu_management \
         --batch_size 1 \
         --device cuda

LM-Evaluation-Harness on ONNX Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. LM-Evaluation-Harness on pretrained, ONNX Exported LLM: Example with
   ``Llama2-7b-hf``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --import_file_format onnx \
         --import_model_dir [path to Llama-2-7b-hf ONNX model] \
         --model hf \
         --tasks mmlu_management \
         --batch_size 1 \
         --device cpu

2. LM-Evaluation-Harness on Quark Quantized, ONNX Exported LLM: Example with
   ``Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx``:

.. code:: bash

     python llm_eval.py \
         --model_args pretrained=meta-llama/Llama-2-7b-hf \
         --import_file_format onnx_format \
         --import_model_dir [path to Llama-2-7b-chat-hf-awq-int4-asym-gs128-onnx model] \
         --model hf \
         --tasks mmlu_management \
         --batch_size 1 \
         --device cpu

Other Arguments
---------------

1. Set ``--multi_gpu`` for multi-gpu support.
2. Set ``dtype`` by ``model_args dtype=float32`` to change model dtype.
3. See a list of supported args by LM-Evaluation-Harness
   `here <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md>`__.
   A few noteworthy ones are ``--limit`` to limit the number of samples evaluated, ``--num_fewshot`` to specify number of examples in fewshot
   setup.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
