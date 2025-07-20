Language Model Post Training Quantization (PTQ) Using Quark
===========================================================

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_ptq``.

This document provides examples of post training quantizing (PTQ) and exporting the language models (such as OPT and Llama) using Quark. For evaluation of quantized model, refer to :doc:`Model Evaluation <example_quark_torch_llm_eval>`.

Supported Models
----------------

.. list-table:: Supported Models
   :widths: 40 10 10 10 10 10 10
   :header-rows: 1

   * - Model Name
     - FP8①
     - INT②
     - MX③
     - AWQ/GPTQ(INT)④
     - SmoothQuant
     - Rotation
   * - meta-llama/Llama-2-\*-hf ⑤
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3-\*B(-Instruct)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3.1-\*B(-Instruct)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3.2-\*B(-Instruct)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3.2-\*B-Vision(-Instruct) ⑥
     - ✓
     - ✓
     -
     -
     -
     -
   * - facebook/opt-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - EleutherAI/gpt-j-6b
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - THUDM/chatglm3-6b
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen1.5-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen1.5-MoE-A2.7B
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen2-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - microsoft/phi-2
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - microsoft/Phi-3-mini-\*k-instruct
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - microsoft/Phi-3.5-mini-instruct
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - mistralai/Mistral-7B-v0.1
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - mistralai/Mixtral-8x7B-v0.1
     - ✓
     - ✓
     -
     -
     -
     -
   * - hpcai-tech/grok-1
     - ✓
     - ✓
     -
     - ✓
     -
     -
   * - CohereForAI/c4ai-command-r-plus-08-2024
     - ✓
     -
     -
     -
     -
     -
   * - CohereForAI/c4ai-command-r-08-2024
     - ✓
     -
     -
     -
     -
     -
   * - CohereForAI/c4ai-command-r-plus
     - ✓
     -
     -
     -
     -
     -
   * - CohereForAI/c4ai-command-r-v01
     - ✓
     -
     -
     -
     -
     -
   * - databricks/dbrx-instruct
     - ✓
     -
     -
     -
     -
     -
   * - deepseek-ai/deepseek-moe-16b-chat
     - ✓
     -
     -
     -
     -
     -

.. note::

   - FP8 means ``OCP fp8_e4m3`` data type quantization.
   - INT includes INT8, UINT8, INT4, UINT4 data type quantization
   - MX includes OCP data type MXINT8, MXFP8E4M3, MXFP8E5M2, MXFP4, MXFP6E3M2, MXFP6E2M3.
   - GPTQ only supports QuantScheme as 'PerGroup' and 'PerChannel'.
   - ``\*`` represents different model sizes, such as ``7b``.
   - meta-llama/Llama-3.2-\*B-Vision models only quantize language parts.

Preparation
-----------

For Llama2 models, download the HF Llama2 checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta. For additional details, see the Llama2 page on Huggingface. Upon obtaining permission, download the checkpoint to the `[llama checkpoint folder]`.

Quantization & Export Scripts & Import Scripts
----------------------------------------------

You can run the following Python scripts in the current path. Here we use Llama as an example.

.. note::

   - To avoid memory limitations, GPU users can add the `--multi_gpu` argument when running the model on multiple GPUs.
   - CPU users should add the `--device cpu` argument.

Recipe 1: Evaluation of Llama Float16 Model without Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --skip_quantization

Recipe 2: FP8 (OCP fp8_e4m3) Quantization & Json_SafeTensors_Export with KV Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to export the autofp8 format for use in downstream libraries such as vLLM, please add '--custom_mode fp8'.

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_fp8_a_fp8 \
                             --kv_cache_dtype fp8 \
                             --num_calib_data 128 \
                             --model_export hf_format

Recipe 3: INT Weight-Only Quantization & Json_SafeTensors_Export with AWQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to export the autoawq format, please add '--custom_mode awq'.

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_int4_per_group_sym \
                             --num_calib_data 128 \
                             --quant_algo awq \
                             --dataset pileval_for_awq_benchmark \
                             --seq_len 512 \
                             --model_export hf_format

Recipe 4: INT Static Quantization & Json_SafeTensors_Export (on CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_int8_a_int8_per_tensor_sym \
                             --num_calib_data 128 \
                             --device cpu \
                             --model_export hf_format

Recipe 5: Quantization & GGUF_Export with AWQ (W_uint4 A_float16 per_group asymmetric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_uint4_per_group_asym \
                             --quant_algo awq \
                             --num_calib_data 128 \
                             --group_size 32 \
                             --model_export gguf

Recipe 6: MX Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype microscaling, abbreviated as MX. Use the following command to quantize the model to datatype MX:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mxfp8 \
                             --num_calib_data 32 \
                             --group_size 32

The command above is weight-only quantization. If you want activations to be quantized as well, use the command below:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mxfp8_a_mxfp8 \
                             --num_calib_data 32 \
                             --group_size 32

Recipe 7: BFP16 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype BFP16 (Block Floating Point 16 bits). Use the following command to quantize the model to datatype BFP16:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_bfp16 \
                             --num_calib_data 16

The command above is weight-only quantization. If you want activations to be quantized as well, use the command below:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_bfp16_a_bfp16 \
                             --num_calib_data 16

Recipe 8: MX6 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype MX6. Use the following command to quantize the model to datatype MX6:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mx6 \
                             --num_calib_data 16

The command above is weight-only quantization. If you want activations to be quantized as well, use the command below:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mx6_a_mx6 \
                             --num_calib_data 16

Recipe 9: Two-Stage Quantization: 1st Stage FP4 Per-Group & 2nd Stage FP8 Per-Tensor for Scale of 1st Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the two-stage quantization scheme. The first stage is FP4 Per-Group and the second stage is FP8 Per-Tensor quantization for scale of 1st Stage.

  .. code-block:: bash

    python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                              --output_dir output_dir \
                              --quant_scheme w_fp4_scale_fp8 \
                              --num_calib_data 16

The command above is weight-only quantization. If you want activations to be quantized as well, use the command below:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_fp4_a_fp4_scale_fp8 \
                             --num_calib_data 16

Recipe 10: MOE Model Experts Weights Second Step Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For MOE structure model, Quark supports second step quantization for weights in the expert layers. Use the following command to quantize the model:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [moe structure model checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_fp8_a_fp8 \
                             --kv_cache_dtype fp8 \
                             --moe_experts_second_step_config w_int4_per_channel_sym \
                             --num_calib_data 16

Recipe 11: Import Quantized Model & Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantized model can be imported and evaluated:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --import_model_dir [path to quantized model] \
                             --model_reload \
                             --import_file_format hf_format

.. note::

   Exporting quantized MX6 model is not supported yet.

Tutorial: Running a Model Not on the Supported List
---------------------------------------------------

For a new model that is not listed in Quark, you need to modify some relevant files. Follow these steps:

1. Add the model type to `MODEL_NAME_PATTERN_MAP` in `get_model_type` function in `quantize_quark.py`.

   `MODEL_NAME_PATTERN_MAP` describes model type, which is used to configure the `quant_config` for the models. You can use part of the model's HF-ID as the key of the dictionary, and the lowercase version of this key as the value.

   .. code-block:: python

      def get_model_type(model: nn.Module) -> str:
          MODEL_NAME_PATTERN_MAP = {
              "Llama": "llama",
              "OPT": "opt",
              ...
              "Cohere": "cohere",  # <---- Add code HERE
          }
          for k, v in MODEL_NAME_PATTERN_MAP.items():
              if k.lower() in type(model).__name__.lower():
                  return v

2. Customize tokenizer for your model in `get_tokenizer` function in `quantize_quark.py`.

   For the most part, the `get_tokenizer` function is applicable. But for some models, such as `CohereForAI/c4ai-command-r-v01`, `use_fast` can only be set to `True` (as of transformers-4.44.2). You can customize the tokenizer by referring to your model's Model card on Hugging Face and `tokenization_auto.py` in transformers.

   .. code-block:: python

      def get_tokenizer(ckpt_path: str, max_seq_len: int = 2048, model_type: Optional[str] = None) -> AutoTokenizer:
          print(f"Initializing tokenizer from {ckpt_path}")
          use_fast = True if model_type == "grok" or model_type == "cohere" else False
          tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                                    model_max_length=max_seq_len,
                                                    padding_side="left",
                                                    trust_remote_code=True,
                                                    use_fast=use_fast)

3. [Optional] For some layers you don't want to quantize, add them to `MODEL_NAME_EXCLUDE_LAYERS_MAP` in `configuration_preparation.py`.

   If you are quantizing an MoE model, the gate layers do not need to be quantized, or there are other layers that you do not want to quantize. You can add `model_type` and excluding layer name to `MODEL_NAME_EXCLUDE_LAYERS_MAP`.

   .. code-block:: python

      MODEL_NAME_EXCLUDE_LAYERS_MAP = {
          "llama": ["lm_head"],
          "opt": ["lm_head"],
          ...
          "cohere": ["lm_head"],  # <---- Add code HERE
      }

4. [Optional] If quantizing `kv_cache`, add the names of kv layers to `MODEL_NAME_KV_LAYERS_MAP` in `configuration_preparation.py`.

   When quantizing `kv_cache`, add `model_type` and kv layers name to `MODEL_NAME_KV_LAYERS_MAP`.

   .. code-block:: python

      MODEL_NAME_KV_LAYERS_MAP = {
          "llama": ["*k_proj", "*v_proj"],
          "opt": ["*k_proj", "*v_proj"],
          ...
          "cohere": ["*k_proj", "*v_proj"],  # <---- Add code HERE
      }

5. [Optional] If using GPTQ, SmoothQuant, and AWQ, add `awq_config.json` and `gptq_config.json` for the model.

   Quark relies on `awq_config.json` and `gptq_config.json` to execute GPTQ, SmoothQuant, and AWQ.

   Create a model directory named after the `model_type` under `Quark/examples/torch/language_modeling/models` and create `awq_config.json` and `gptq_config.json` in this directory.

   For GPTQ:

   The config file should be named `gptq_config.json`. You should collate all linear layers in decoder layers and put them in the `inside_layer_modules` list and put the decoder layers name in the `model_decoder_layers` list.

   For SmoothQuant and AWQ:

   SmoothQuant and AWQ use the same file named `awq_config.json`. In general, for each decoder layer, you need to process four parts (`linear_qkv`, `linear_o`, `linear_mlp_fc1`, `linear_mlp_fc2`). You can refer to existing configurations for guidance.

Tutorial: Generating AWQ Configuration Automatically (Experimental)
-------------------------------------------------------------------

We provide a script `awq_auto_config_helper.py` to simplify user operations by quickly identifying modules compatible with the "AWQ" and "SmoothQuant" algorithms within the model through `torch.compile`.

Installation
------------

This script requires PyTorch version 2.4 or higher.

Usage
-----

The `MODEL_DIR` variable should be set to the model name from Hugging Face, such as `facebook/opt-125m`, `Qwen/Qwen2-0.5B`, or `EleutherAI/gpt-j-6b`.

To run the script, use the following command:

.. code-block:: bash

   MODEL_DIR="your_model"
   python awq_auto_config_helper.py --model_dir "${MODEL_DIR}"
