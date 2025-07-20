Integration with AMD Pytorch-light (APL)
========================================

.. note::

   For information on accessing Quark PyTorch examples, refer to `Accessing PyTorch Examples <pytorch_examples>`_.
   This example and the relevant files are available at ``/torch/extensions/pytroch_light``.

Introduction
------------

This topic demonstrates the integration of **APL** into Quark. APL is a lightweight model optimization library based on PyTorch, designed primarily for developers. **APL** is AMD's internal quantization framework; external users need to request access. Ensure **APL** is installed before running this example.

**APL** supports a variety of quantization methods and advanced quantization data types. ``Quark`` provides a user-friendly interface, allowing users to easily leverage these quantization techniques. This example combines the strengths of both frameworks, enabling users to invoke **APL** through Quark's interface. We have prepared three examples that demonstrate the use of the BFP16, INTK, and BRECQ quantization schemes of **APL** via Quark's interface.

Example 1
---------

In this example, we use the Llama2 model and call **APL** to perform the INTK model quantization.

- Model: ``llama2 7b``
- Calibration method: ``minmax``
- Quantization data type: ``int8``

**Replace Operations**:

- Replace ``nn.Linear`` with ``pytorchlight.nn.linear_layer.LLinear``

**Run Script**

.. code-block:: bash

   model_path={your `llama-2-7b` model path}
   dataset_path={your data path}

   python quantize_quark.py \
       --model llama-7b \
       --model_path ${model_path} \
       --seqlen 4096 \
       --dataset_path ${dataset_path} \
       --eval

Example 2
---------

In this example, we use the OPT-125M model and call **APL** to perform BFP16 model quantization. We support the quantization of ``nn.Linear``, ``nn.LayerNorm``, and ``nn.Softmax`` through **APL**.

- Model: ``opt-125m``
- Calibration method: ``minmax``
- Quantization data type: ``bfp16``

**Replace Operations**:

- Replace ``nn.Linear`` with ``pytorchlight.nn.linear_layer.LLinear``
- Replace ``nn.LayerNorm`` with ``pytorchlight.nn.normalization_layer.LLayerNorm``
- Replace ``nn.Softmax`` with ``pytorchlight.nn.activation_layer.LSoftmax``

**Run Script**

.. code-block:: bash

   model_path={your `opt-125m` model path}
   dataset_path={your data path}

   python quantize_quark.py \
       --model opt-125m \
       --model_path ${model_path} \
       --seqlen 4096 \
       --qconfig 0 \
       --eval \
       --qscale_type fp32 \
       --dataset_path ${dataset_path} \
       --example bfp16

Example 3
---------

This example demonstrates how to use the ``brecq`` algorithm through ``Quark`` to call **APL**.

- Model: ``opt-125m``
- Calibration method: ``minmax``
- Quantization data type: ``int8``

.. code-block:: bash

   model_path={your `opt-125m` model path}
   dataset_path={your data path}

   export CUDA_VISIBLE_DEVICES=0,5,6;
   python quantize_quark.py \
       --model opt-125m \
       --model_path ${model_path} \
       --seqlen 1024 \
       --eval \
       --example brecq \
       --dataset_path ${dataset_path}
