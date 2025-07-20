Block Floating Point (BFP) Example
==================================

.. note::

   For information on accessing Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <onnx_examples>`.
   This example and the relevant files are available at ``/onnx/accuracy_improvement/BFP``.

This is an example of quantizing a `mobilenetv2_050.lamb_in1k` model using the ONNX quantizer of Quark with BFP16.
Int8 quantization performs poorly on the model, but BFP16 and ADAQUANT can significantly mitigate the quantization loss.

Block Floating Point (BFP) quantization computational complexity by grouping numbers to share a common exponent, preserving accuracy efficiently.
BFP has both reduced storage requirements and high quantization precision.

The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `BFP16 Quantization <#bfp16-quantization>`__
-  `BFP16 Quantization with ADAQUANT <#bfp16-quantization-with-adaquant>`__
-  `Evaluation <#evaluation>`__


Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r ../utils/requirements.txt


Prepare model
-------------

Export onnx model from mobilenetv2_050.lamb_in1k torch model. The corresponding model link is https://huggingface.co/timm/mobilenetv2_050.lamb_in1k:

::

   mkdir models && python ../utils/export_onnx.py mobilenetv2_050.lamb_in1k

Prepare data
------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access
to ImageNet (ILSVRC) 2012 which is the most commonly used subset of
ImageNet. This dataset spans 1000 object classes and contains 50,000
validation images.

If you already have an ImageNet datasets, you can directly use your
dataset path.

To prepare the test data, please check the download section of the main
website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You
need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

::

   mkdir val_data && tar -xzf val_images.tar.gz -C val_data
   python ../utils/prepare_data.py val_data calib_data

The storage format of the val_data of the ImageNet dataset organized as
follows:

- val_data

   - n01440764

      - ILSVRC2012_val_00000293.JPEG
      - ILSVRC2012_val_00002138.JPEG
      - …

   - n01443537

      - ILSVRC2012_val_00000236.JPEG
      - ILSVRC2012_val_00000262.JPEG
      - …

   - …
   - n15075141

      - ILSVRC2012_val_00001079.JPEG
      - ILSVRC2012_val_00002663.JPEG
      - …

The storage format of the calib_data of the ImageNet dataset organized
as follows:

- calib_data

   - n01440764

      - ILSVRC2012_val_00000293.JPEG

   - n01443537

      - ILSVRC2012_val_00000236.JPEG

   - …
   - n15075141

      - ILSVRC2012_val_00001079.JPEG

BFP16 Quantization
------------------

The quantizer takes the float model and produce a BFP16 quantized model.

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config BFP16

This command will generate a BFP16 quantized model under the **models**
folder, which was quantized by BFP16 configuration.

BFP16 Quantization with ADAQUANT
--------------------------------

The quantizer takes the float model and produce a BFP16 quantized model with
ADAQUANT.

Note: If the model has dynamic shapes, you need to convert the model to fixed shapes before performing ADAQUANT.

::

   python -m  quark.onnx.tools.convert_dynamic_to_fixed  --fix_shapes 'input:[1,3,224,224]' models/mobilenetv2_050.lamb_in1k.onnx  models/mobilenetv2_050.lamb_in1k_fix.onnx

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k_fix.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_adaquant_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config BFP16_ADAQUANT

If the GPU is available in your environment, you can accelerate the training process by configuring parameter 'device' as 'rocm' or 'cuda'.

::

   python quantize_model.py --model_name mobilenetv2_050.lamb_in1k \
                            --input_model_path models/mobilenetv2_050.lamb_in1k_fix.onnx \
                            --output_model_path models/mobilenetv2_050.lamb_in1k_adaquant_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config BFP16_ADAQUANT \
                            --device cuda

This command will generate a BFP16 quantized model under the **models**
folder, which was quantized by BFP16 configuration with ADAQUANT.

Evaluation
----------

Test the accuracy of the float model on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k.onnx

Test the accuracy of the BFP16 quantized model on ImageNet
val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_quantized.onnx

If want to run faster with GPU support, you can also execute the following command:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_quantized.onnx --gpu

Test the accuracy of the BFP16 quantized model with ADAQUANT on ImageNet val
dataset:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_adaquant_quantized.onnx

If want to run faster with GPU support, you can also execute the following command:

::

   python ../utils/onnx_validate.py val_data --model-name mobilenetv2_050.lamb_in1k --batch-size 1 --onnx-input models/mobilenetv2_050.lamb_in1k_adaquant_quantized.onnx --gpu

Quantization Results
--------------------

+-------+-------------------+---------------------+-------------------+
|       | Float Model       | Quantized Model     | Quantized Model   |
|       |                   | without ADAQUANT    | with ADAQUANT     |
+=======+===================+=====================+===================+
| Model | 8.7 MB            | 8.4 MB              | 8.4 MB            |
| Size  |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 65.424 %          | 60.806 %            | 64.652 %          |
| rec@1 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 85.788 %          | 82.648 %            | 85.278 %          |
| rec@5 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+

.. note:: Different execution devices can lead to minor variations in the
          accuracy of the quantized model.


.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
