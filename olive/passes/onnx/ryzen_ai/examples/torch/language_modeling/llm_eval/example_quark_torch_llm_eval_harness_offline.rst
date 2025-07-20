LM-Evaluation-Harness (Offline)
===============================

We provide a multi-step flow to run lm-evaluation-harness metrics offline for ONNX models. Offline mode is used to evaluate models generations on specific hardware (i.e., NPUs). The offline mode is invoked through ``llm_eval.py --mode offline``. Currently, only the below generation tasks are supported in offline mode.

Supported Tasks
---------------

[``gsm8k``, ``tinyGSM8k``]

Step-by-Step Process
--------------------

Below are the steps on how to use the offline mode.
Please make sure ``--num_fewshot`` is set to 0 to allow for fair comparisons from
OGA model generations.


1. Retrieve dataset from LM-Evaluation-Harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--retrieve_dataset`` to save dataset inputs.json and references.json.
Example shown with all 100 samples of tinyGSM8k:

.. code-block:: bash

     python llm_eval.py \
         --mode offline \
         --retrieve_dataset \
         --tasks tinyGSM8k \
         --num_fewshot 0

If you do not want entire dataset, use ``--limit <NUM>`` to specify number of data samples.
The output of ``--retrieve_dataset`` will be 4 files: .json and .txt files for both input and references.
You can view these four files in ``example_files/``:

* ``example_files/tinyGSM8k_inputs_limit-None.json`` - input samples for tinyGSM8k that is used in step 3.
* ``example_files/tinyGSM8k_references_limit-None.json`` - references used internally by lm-evaluation-harness in step 5
* ``example_files/tinyGSM8k_references_limit-None.txt`` - inputs as .txt file, showing requirement of <EOR> delimiter between examples
* ``example_files/tinyGSM8k_inputs_limit-None.txt``- references as .txt file, showing requirement of <EOR> delimiter between examples

If you choose to ingest .txt input files as opposed to .json files for step 3, ensure each sample is separated by an <EOR> delimiter.

2. Export Pretrained Model-Of-Interest to ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use OGA Model Builder to save ONNX Pretrained Model.
See `here <https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python>`_ for how to use OGA Model Builder.

3. Retrieve OGA references for pre-trained ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--oga_references`` to save the OGA references for a particular pre-trained model.
Example shown with 20 samples of gsm8k for pre-trained Phi3.5-mini-instruct ONNX Model:

There are 3 cases supported for OGA generation:

* ``default``: Default OGA generation.
* ``psu_prompt_eos_stop``: Inputs are preprended with a prompt for tinyGSM8k and early stopping based on EOS.
* ``psu_prompt``: Inputs are preprended with the PSU prompt for tinyGSM8k.

All cases require offline HW models to use same generation conditions to see comparable results.
Please see ``oga_generation()`` in ``utilities.py``.

Use ``--oga_references`` to save the OGA references for a particular pretrained model.
Example shown with all samples of tinyGSM8k for pretrained Phi3.5-mini-instruct ONNX Model
using default case:

.. code-block:: bash

     python llm_eval.py \
         --mode offline \
         --oga_references \
         --inputs [path to inputs.json] \
         --import_model_dir [path to Phi3.5-mini-instruct ONNX Model] \
         --import_file_format onnx_format \
         --tasks tinyGSM8k \
         --num_fewshot 0 \
         --case default \
         --model_name Phi3.5-mini-instruct-onnx \
         --eor "<EOR>"

4. Get Baseline Evaluation Scores on Pretrained ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--eval_mode`` to compare the pretrained model's references to the dataset references.
Example shown with comparing Phi3.5-mini-instruct ONNX model references to tinyGSM8k references.

.. code-block:: bash

     python llm_eval.py \
         --mode offline \
         --eval_mode \
         --outputs_path [path to Phi3.5-mini-instruct OGA references.txt] \
         --tasks tinyGSM8k \
         --num_fewshot 0 \
         --eor "<EOR>"

5. Evaluate an optimized ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--eval_mode`` to compare an optimized model to the dataset references.
Example shown with comparing a quantized Phi3.5-mini-instruct ONNX model predictions to tinyGSM8k references.

.. code-block:: bash

     python llm_eval.py \
         --mode offline \
         --eval_mode \
         --outputs_path [path to quantized model predictions.txt] \
         --tasks tinyGSM8k \
         --num_fewshot 0 \
         --eor "<EOR>"

Note: predictions.txt should follow the same format as references.txt from (4). This means, each model output must
be separated by a end-of-response delimiter such as ``"<EOR>"``. See example below of the formatting:

.. code-block::

    This would be the first model output.
    <EOR>
    This would be the second model output
    <EOR>

6. Verify Offline Mode Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the sample tinyGSM8k predictions.txt file and run ``--eval mode``. You should see identical results:

1. Running eval using ``default`` OGA generation: ``example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_default.txt``

.. code-block:: bash

    python llm_eval.py \
        --mode offline \
        --eval_mode \
        --outputs_path example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_default.txt \
        --tasks tinyGSM8k \
        --num_fewshot 0 \
        --eor "<EOR>"

.. code-block::

    -------OUTPUT------
    |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |---------|------:|----------------|-----:|-----------|---|-----:|---|------|
    |tinyGSM8k|      0|flexible-extract|     0|exact_match|↑  |0.2907|±  |   N/A|
    |         |       |strict-match    |     0|exact_match|↑  |0.0055|±  |   N/A|

2. Running eval using OGA generation with ``psu_prompt_eos_stop``: ``example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_psu_prompt_eos_stop.txt``

.. code:: bash

    python llm_eval.py \
        --mode offline \
        --eval_mode \
        --outputs_path example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_psu_prompt_eos_stop.txt \
        --tasks tinyGSM8k \
        --num_fewshot 0 \
        --eor "<EOR>"

.. code-block::

    -------OUTPUT------
    |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |---------|------:|----------------|-----:|-----------|---|-----:|---|------|
    |tinyGSM8k|      0|flexible-extract|     0|exact_match|↑  |0.6421|±  |   N/A|
    |         |       |strict-match    |     0|exact_match|↑  |0.0737|±  |   N/A|

3. Running eval using OGA generation with ``psu_prompt``: ``example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_psu_prompt.txt``

.. code:: bash

     python llm_eval.py \
         --mode offline \
         --eval_mode \
         --outputs_path example_files/deepseek-qwen1.5B_tinyGSM8k_limit-None_psu_prompt.txt \
         --tasks tinyGSM8k \
         --num_fewshot 0 \
         --eor "<EOR>"

.. code-block::

    -------OUTPUT------
    |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |---------|------:|----------------|-----:|-----------|---|-----:|---|------|
    |tinyGSM8k|      0|flexible-extract|     0|exact_match|↑  |0.6421|±  |   N/A|
    |         |       |strict-match    |     0|exact_match|↑  |0.0737|±  |   N/A|

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
