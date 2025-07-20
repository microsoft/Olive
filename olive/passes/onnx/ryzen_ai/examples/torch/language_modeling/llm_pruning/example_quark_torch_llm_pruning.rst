Pruning
=======

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_pruning``.

This topic contains examples of pruning language models (such as OPT and Llama) using Quark.

Supported Models
----------------

+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| Model Name                            | Model Size | Pruning Rate | Pruned Model Size | Before Pruning PPL On Wiki2 | After Pruning PPL On Wiki2 |
+=======================================+============+==============+===================+=============================+============================+
| mistralai/Mixtral-8x7B-Instruct-v0.1  | 46.7B      | 9.4838%      | 42.2B             | 4.1370                      | 5.1195                     |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| CohereForAI/c4ai-command-r-08-2024    | 32.3B      | 7.4025%      | 29.9B             | 4.5081                      | 6.3794                     |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| Qwen/Qwen2.5-14B-Instruct             | 14.8B      | 7.0284%      | 13.7B             | 5.6986                      | 7.5994                     |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| meta-llama/Meta-Llama-3-8B            | 8.0B       | 6.8945%      | 7.5B              | 6.1382                      | 8.0755                     |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| meta-llama/Llama-2-7b-hf              | 6.7B       | 6.7224%      | 6.2B              | 5.4721                      | 6.2462                     |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| facebook/opt-6.7b                     | 6.7B       | 7.5651%      | 6.2B              | 10.8602                     | 11.8958                    |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| THUDM/chatglm3-6b                     | 6.2B       | 7.7590%      | 5.6B              | 29.9560                     | 36.0010                    |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+
| microsoft/Phi-3.5-mini-instruct       | 3.8B       | 5.9274%      | 3.6B              | 6.1959                      | 7.8074                     |
+---------------------------------------+------------+--------------+-------------------+-----------------------------+----------------------------+

Preparation
-----------

For Llama2 models, download the HF Llama2 checkpoint. Access the Llama2 models checkpoint by submitting a permission request to Meta. For additional details, see the Llama2 page on Huggingface. Upon obtaining permission, download the checkpoint to the ``[llama2_checkpoint_folder]``.

Pruning Scripts
---------------

Run the following Python scripts in the ``examples/torch/language_modeling/llm_pruning`` path. Use Llama2-7b as an example.

.. note::

    - To avoid memory limitations, GPU users can add the ``--multi_gpu`` argument when running the model on multiple GPUs.
    - CPU users should add the ``--device cpu`` argument.

Recipe 1: Evaluation of Llama2 Float16 Model without Pruning
------------------------------------------------------------

.. code-block:: bash

    python3 main.py --model_dir [llama2 checkpoint folder] \
                             --skip_pruning

Recipe 2: Pruning Model and Saved to Safetensors
------------------------------------------------

.. code-block:: bash

    python3 main.py --model_dir [llama2 checkpoint folder] \
                             --pruning_algo "osscar" \
                             --num_calib_data 128 \
                             --save_pruned_model \
                             --save_dir save_dir
