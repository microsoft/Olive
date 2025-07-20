.. raw:: html

   <!-- omit in toc -->

Quark ONNX Quantization Example
===============================

This folder contains an example of quantizing a llama2-7B model using the ONNX quantizer of Quark. It also shows how to use the QuaRot algorithm.

Introduction
------------

This example mainly shows how to use QuaRot [https://arxiv.org/abs/2404.00456] algorithm during quantization. QuaRot is proposed to harmonize the outliers within the activations before MatMul/Gemm.
The main idea for QuaRot is to insert Hadamard transformation pairs into activations, hence projecting activations to Hadamard domain.
This projection can make discrete energy become concentrated, or make concentrated energy become discrete.
Due to the discrete distribution of activation, the distribution after the Hadamard transform becomes more concentrated, thereby mitigating the outlier situation and relieving activation quantization error.

QuaRot set up 4 types of rotation pairs (R1/R2/R3/R4), which are further discussed in SpinQuant [https://arxiv.org/abs/2405.16406]. Currently, we only support R1 rotation.

Guidelines
----------

The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Quantization without QuaRot <#quantization-without-quarot>`__
-  `Quantization with QuaRot <#quantization-with-quarot>`__
-  `Quantization with QuaRot and SmoothQuant <#quantization-with-quarot-and-smoothquant>`__
-  `Evaluation <#evaluation>`__

Pip requirements
----------------

Install the necessary python packages:

.. code-block:: bash

   python -m pip install -r requirements.txt

Prepare model
-------------

Download the HF Llama-2-7b-hf checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta.
For additional details, see the `Llama2 page on Huggingface <https://huggingface.co/docs/transformers/main/en/model_doc/llama2>`__. Upon obtaining permission, download the checkpoint to the **Llama-2-7b-hf** folder.

Export oga model from Llama-2-7b-hf torch model:

.. code-block:: bash

   mkdir oga_fp32_model && python export2oga.py --model_path ./Llama-2-7b-hf --output_path oga_fp32_model

Quantization without QuaRot
---------------------------

The quantizer takes the float model and produces a quantized model without QuaRot.

.. code-block:: bash

   mkdir quantized_models
   cp -r oga_fp32_model/*.json quantized_models
   cp oga_fp32_model/tokenizer.model quantized_models
   python quantize_model.py --input_model_path oga_fp32_model/model.onnx \
                            --output_model_path quantized_models/model.onnx \
                            --num_calib_data 64 \
                            --use_moving_average \
                            --config INT8_TRANSFORMER_DEFAULT


This command will generate a quantized model under the **quantized_models** folder, which was quantized by Int8 configuration for transformer-based models.

Quantization with QuaRot
------------------------

The quantizer takes the float model and produce a quantized model with QuaRot.

.. code-block:: bash

   mkdir rotated_quantized_models
   cp -r oga_fp32_model/*.json rotated_quantized_models
   cp oga_fp32_model/tokenizer.model rotated_quantized_models
   python quantize_model.py --input_model_path oga_fp32_model/model.onnx \
                            --output_model_path rotated_quantized_models/model.onnx \
                            --include_rotation \
                            --r_config_path rotation_config.json \
                            --hidden_size 4096 \
                            --num_calib_data 64 \
                            --use_moving_average \
                            --config INT8_TRANSFORMER_DEFAULT

This command will generate a quantized model under the **rotated_quantized_models** folder, which was quantized by Int8 configuration for transformer-based models with QuaRot.

Quantization with QuaRot and SmoothQuant
----------------------------------------

QuaRot and SmoothQuant can be used in conjunction to achieve better optimization results.
The quantizer takes the float model and produce a quantized model with QuaRot and SmoothQuant.

.. code-block:: bash

   mkdir rotated_smoothed_quantized_models
   cp -r oga_fp32_model/*.json rotated_smoothed_quantized_models
   cp oga_fp32_model/tokenizer.model rotated_smoothed_quantized_models
   python quantize_model.py --input_model_path oga_fp32_model/model.onnx \
                            --output_model_path rotated_smoothed_quantized_models/model.onnx \
                            --include_rotation \
                            --r_config_path rotation_config.json \
                            --hidden_size 4096 \
                            --include_sq \
                            --sq_alpha 0.5 \
                            --num_calib_data 64 \
                            --use_moving_average \
                            --config INT8_TRANSFORMER_DEFAULT

This command will generate a quantized model under the **rotated_smoothed_quantized_models** folder, which was quantized by Int8 configuration for transformer-based models with QuaRot and SmoothQuant.

Evaluation
----------

Test the PPL of the float model on wikitext2.raw:

.. code-block:: bash

   python oga_validate.py --model_name_or_path oga_fp32_model/ --do_onnx_eval --block_size 2048

Test the PPL of the quantized model without QuaRot:

.. code-block:: bash

   python oga_validate.py --model_name_or_path quantized_models/ --do_onnx_eval --block_size 2048

Test the PPL of the quantized model with QuaRot:

.. code-block:: bash

   python oga_validate.py --model_name_or_path rotated_quantized_models/ --do_onnx_eval --block_size 2048

Test the PPL of the quantized model with QuaRot and SmoothQuant:

.. code-block:: bash

   python oga_validate.py --model_name_or_path rotated_smoothed_quantized_models/ --do_onnx_eval --block_size 2048

+-------+--------------------+---------------------+---------------------+-------------------------+
|       | Float Model        | Quantized Model     | Quantized Model     | Quantized Model with    |
|       |                    | without QuaRot      | with QuaRot         | QuaRot and SmoothQuant  |
+=======+====================+=====================+=====================+=========================+
| Model | 26.0 G             | 6.7 G               | 6.7 G               | 6.8 G                   |
| Size  |                    |                     |                     |                         |
+-------+--------------------+---------------------+---------------------+-------------------------+
| PPL   | 5.63               | 16.09               | 11.02               | 6.10                    |
+-------+--------------------+---------------------+---------------------+-------------------------+

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
