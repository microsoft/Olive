Quantizating a model with GPTQ
==============================

This folder contains an example of quantizing a opt-125m model using the ONNX quantizer of Quark. It also shows how to use the GPTQ algorithm.

The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Quantization without GPTQ <#quantization-without-gptq-quant>`__
-  `Quantization with GPTQ <#quantization-with-gptq-quant>`__
-  `Evaluation <#evaluation>`__

Pip Requirements
^^^^^^^^^^^^^^^^

Install the necessary python packages:

::

   python -m pip install -r requirements.txt

Prepare Model
^^^^^^^^^^^^^

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

Export onnx model from opt-125m torch model:

::

   mkdir models && optimum-cli export onnx --model ./opt-125m --task text-generation ./models/

Quantization Without GPTQ
^^^^^^^^^^^^^^^^^^^^^^^^^

The quantizer takes the float model and produces a quantized model without GPTQ.

::

   cp -r models quantized_models && rm quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path quantized_models/quantized_model.onnx \
                            --config INT8_TRANSFORMER_DEFAULT

This command will generate a quantized model under the **quantized_models** folder, which was quantized by Int8 configuration for transformer-based models.

Quantization With GPTQ
^^^^^^^^^^^^^^^^^^^^^^

The quantizer takes the float model and produces a quantized model with QDQ GPTQ (8-bits).

::

   cp -r models gptq_quantized_models && rm gptq_quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path gptq_quantized_models/gptq_quantized_model.onnx \
                            --config INT8_TRANSFORMER_DEFAULT \
                            --use_gptq

This command will generate a quantized model under the **gptq_quantized_models** folder, which was quantized by Int8 configuration for transformer-based models with 8-bits GPTQ Quant.

The quantizer takes the float model and produces a quantized model with MatMulNBits GPTQ (4-bits).

::

   cp -r models gptq_quantized_models && rm gptq_quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path gptq_quantized_models/gptq_quantized_model.onnx \
                            --config MATMUL_NBITS

This command will generate a quantized model under the **gptq_quantized_models** folder, which was quantized by MATMUL_NBITS configuration for transformer-based models with 4-bits GPTQ Quant.

Evaluation
^^^^^^^^^^

Test the PPL of the float model on wikitext2.raw:

::

   python onnx_validate.py --model_name_or_path models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model models/ --do_onnx_eval --no_cuda

Test the PPL of the quantized model without GPTQ:

::

   python onnx_validate.py --model_name_or_path quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model quantized_models/ --do_onnx_eval --no_cuda

Test the PPL of the quantized model with GPTQ:

::

   python onnx_validate.py --model_name_or_path gptq_quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model gptq_quantized_models/ --do_onnx_eval --no_cuda

.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized Model without GPTQ (8-bits)
     - Quantized Model with GPTQ (8-bits)
     - Quantized Model with MatMulNBits GPTQ (4-bits)
   * - Model Size
     - 480 MB
     - 384 MB
     - 384 MB
     - 406 MB
   * - PPL
     - 27.0317
     - 28.6846
     - 27.5734
     - 30.3604
