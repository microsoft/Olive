Language Model QAT Using Quark and Trainer
===========================================================

This document provides examples of Quantization-Aware Training (QAT) for language models using Quark.

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_qat``.

Supported Models
----------------

+-----------------------------------------+-------------------------------+
| Model Name                              | WEIGHT-ONLY (INT4.g128)       |
+=========================================+===============================+
| microsoft/Phi-3-mini-4k-instruct        | ✓                             |
+-----------------------------------------+-------------------------------+
| THUDM/chatglm3-6b                       | ✓                             |
+-----------------------------------------+-------------------------------+

Preparation
-----------

Please install the required packages before running QAT by executing ``pip install -r requirements.txt``. To evaluate the model, install the necessary dependencies by running ``pip install -r ../llm_eval/requirements.txt``.
If an NCCL timeout error occurs while saving the model during the program's execution, you can try installing the accelerate==1.4.0 version to resolve it.

(Optional) For LLM models, download the Hugging Face checkpoint.

QAT Scripts
-----------

You can run the following Python scripts in the ``examples/torch/language_modeling/llm_qat`` path. Here, Phi-3-mini-4k-instruct is used as an example.



Recipe 1: QAT Finetuning ChatGLM and Export to Safetensors using FSDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    SECONDS=0
    log_file=${log_dir}/llm_qat_${model_name}_finetune.log
    output_dir="./quantized_model/chatglm_6b"
    NUM_GPUS=4
    BATCH_SIZE_PER_GPU=2
    TOTAL_BATCH_SIZE=32
    GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

    FSDP_CONFIG=./fsdp_configs/chatglm_fsdp_config.json
    NUM_EPOCHS=5
    LR=2e-5
    MAX_SEQ_LEN=512

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=${NUM_GPUS} main.py \
                        --fsdp "full_shard auto_wrap" \
                        --fsdp_config ${FSDP_CONFIG} \
                        --model ${MODEL_DIR} \
                        --model_trust_remote_code \
                        --quant_scheme w_uint4_asym \
                        --group_size 128 \
                        --finetune_dataset wikitext \
                        --num_train_epochs ${NUM_EPOCHS} \
                        --learning_rate ${LR} \
                        --finetune_seqlen ${MAX_SEQ_LEN} \
                        --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
                        --per_device_eval_batch_size ${BATCH_SIZE_PER_GPU} \
                        --model_export hf_format \
                        --output_dir $finetune_checkpoint \
                        --model_export_dir ${output_dir} \
                        --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
                        --skip_evaluation 2>&1| tee $log_file
    date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
    TOTAL_TIME=$((TOTAL_TIME+SECONDS))


Recipe 2: Reload and Evaluate QAT Finetuned Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    SECONDS=0
    log_file=${log_dir}/llm_qat_${model_name}_test_finetuned.log
    EVAL_BATCH=4
    export CUDA_VISIBLE_DEVICES=5
    EVAL_TASK=wikitext,winogrande,mmlu
    EVAL_OUTPUT_PATH=./${model_name}_${EVAL_TASK//,/_}_quantized_eval_results
    python main.py \
            --model ${MODEL_DIR} \
            --output_dir $finetune_checkpoint \
            --model_trust_remote_code \
            --skip_finetune \
            --model_reload \
            --import_model_dir $output_dir \
            --eval_result_output_path ${EVAL_OUTPUT_PATH} \
            --per_device_eval_batch_size ${EVAL_BATCH} \
            --eval_task ${EVAL_TASK} 2>&1| tee $log_file
    date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" | tee -a ${log_file}
    TOTAL_TIME=$((TOTAL_TIME+SECONDS))

Recipe 3: Evaluate Original Non-Quantized Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    EVAL_TASK=wikitext,winogrande,mmlu
    EVAL_OUTPUT_PATH=./${model_name}_${EVAL_TASK//,/_}_non_quantized_eval_results
    SECONDS=0
    EVAL_BATCH=4
    log_file=${log_dir}/llm_qat_${model_name}_test_bf16.log
    export CUDA_VISIBLE_DEVICES=4
    python main.py \
            --model ${MODEL_DIR} \
            --output_dir $finetune_checkpoint \
            --model_trust_remote_code \
            --skip_quantization \
            --skip_finetune \
            --eval_result_output_path ${EVAL_OUTPUT_PATH} \
            --per_device_eval_batch_size ${EVAL_BATCH} \
            --eval_task ${EVAL_TASK} 2>&1| tee ${log_file}
    date -ud "@$SECONDS" "+Time elapsed: %H:%M:%S" |tee -a ${log_file}
    TOTAL_TIME=$((TOTAL_TIME+SECONDS))



Results on Phi-3-mini-4k-instruct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+----------------------+----------------------------+-------+------------+
| Model Name       | Wikitext PPL (Quark) | Wikitext PPL (LLM harness) | MMLU  | Winogrande |
+==================+======================+============================+=======+============+
| BF16             |  6.19                | 10.32                      | 68.59 | 74.42      |
+------------------+----------------------+----------------------------+-------+------------+
| QAT Trainer      |  6.21                | 11.51                      | 65.97 | 73.24      |
+------------------+----------------------+----------------------------+-------+------------+


Results on ChatGLM3-6B
~~~~~~~~~~~~~~~~~~~~~~

+------------------+----------------------+----------------------------+-------+------------+
| Model Name       | Wikitext PPL (Quark) | Wikitext PPL (LLM harness) | MMLU  | Winogrande |
+==================+======================+============================+=======+============+
| BF16             |  29.93               | 51.30                      | 50.45 | 62.35      |
+------------------+----------------------+----------------------------+-------+------------+
| QAT Trainer      |  9.84                | 29.97                      | 49.36 | 65.50      |
+------------------+----------------------+----------------------------+-------+------------+



.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
